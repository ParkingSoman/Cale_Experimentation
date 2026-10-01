"""Build models/g12-nest/model.nest: the 12B with 4-, 8- and 16-bit tiers and MLP widths in one file.

Sources:
  plane A, QAT norms   models/qg4 (already in importance order, perm-wanda)
  B, C, original norms models/gemma-4-12B-it-bf16, put in the same neuron order
Plane math: nest/planes.py. File format: nest/nestfile.py.

MLP widths: gate/up rows are in importance order, so a width is a row prefix.
down_proj is stored as column blocks [0, 85%), [85%, 90%), [90%, 100%), each its own
contiguous matrix, so every width reads contiguous bytes.

usage: build_nest.py [OUT] [LAYER]   (default models/g12-nest/model.nest; LAYER: test build of one layer)
Writes OUT and results/build-nest-errors.json (per-table weight error of each tier).
"""
import json
import os
import sys
import time
from collections import defaultdict

import mlx.core as mx
import numpy as np

from nest.nestfile import NestWriter
from nest.planes import A_BITS, A_GROUP, B_BITS, B_GROUP, C_BITS, C_GROUP, make_planes

ORIG = "models/gemma-4-12B-it-bf16"
QG4 = "models/qg4"
OUT = sys.argv[1] if len(sys.argv) > 1 else "models/g12-nest/model.nest"
ONLY = int(sys.argv[2]) if len(sys.argv) > 2 else None
EXTRA_FILES = ["chat_template.jinja", "generation_config.json", "processor_config.json",
               "tokenizer_config.json", "tokenizer.json"]
SKIP = ("embed_audio.",)            # mlx-vlm drops it: the 12B has no audio tower
WIDTHS = {"100": 1.0, "90": 0.90, "85": 0.85}
ROW_CHUNK = 16384                   # word table is built in row chunks

perms = np.load("results/perm-wanda.npz")
qmap = json.load(open(f"{QG4}/model.safetensors.index.json"))["weight_map"]
omap = json.load(open(f"{ORIG}/model.safetensors.index.json"))["weight_map"]
_files = {}


def get(src, wm, key):
    f = f"{src}/{wm[key]}"
    if f not in _files:
        _files[f] = mx.load(f)
    return _files[f][key]


def layer_of(key):
    return int(key.split(".layers.")[1].split(".")[0]) if ".layers." in key else -1


def width_kind(m):
    if ".mlp.gate_proj" in m or ".mlp.up_proj" in m:
        return "rows"
    if ".mlp.down_proj" in m:
        return "cols"
    return "none"


# --- plan every tensor -------------------------------------------------------------
tables = {}
for k in qmap:
    if k.endswith(".scales") and not k.startswith(SKIP):
        m = k[: -len(".scales")]
        out_dim, n_groups = get(QG4, qmap, k).shape
        tables[m] = {"shape": [out_dim, n_groups * A_GROUP], "kind": width_kind(m), "layer": layer_of(m)}
unquant = sorted(k for k in omap if not k.startswith(SKIP)
                 and k[: -len(".weight")] not in tables)
assert set(unquant) <= set(qmap), "QAT build is missing unquantized tensors"
if ONLY is not None:
    tables = {m: t for m, t in tables.items() if t["layer"] == ONLY}
    unquant = [k for k in unquant if layer_of(k) == ONLY]
order = sorted(tables, key=lambda m: (tables[m]["layer"], m))

I = json.load(open(f"{ORIG}/config.json"))["text_config"]["intermediate_size"]
cuts = sorted({int(round(I * w / 64)) * 64 for w in WIDTHS.values()})   # 13056, 13824, 15360
BLOCKS = [0] + cuts
for m in order:
    t = tables[m]
    t["blocks"] = BLOCKS if t["kind"] == "cols" else [0, t["shape"][1]]

specs = []


def plane_specs(m, plane, group, bits):
    t = tables[m]
    out_dim = t["shape"][0]
    bl = t["blocks"]
    for b in range(len(bl) - 1):
        cols = bl[b + 1] - bl[b]
        tag = f"{plane}{b}" if t["kind"] == "cols" else plane
        region = plane
        specs.append((f"{m}|{tag}|weight", region, "uint32", (out_dim, cols * bits // 32)))
        specs.append((f"{m}|{tag}|scales", region, "bfloat16", (out_dim, cols // group)))
        specs.append((f"{m}|{tag}|biases", region, "bfloat16", (out_dim, cols // group)))


for f in EXTRA_FILES:
    specs.append((f"file|{f}", "X", "uint8", (os.path.getsize(f"{QG4}/{f}"),)))
cfg_bytes = open(f"{ORIG}/config.json", "rb").read()
specs.append(("file|config.json", "X", "uint8", (len(cfg_bytes),)))
for m in order:
    plane_specs(m, "A", A_GROUP, A_BITS)
    specs.append((f"{m}|B|row_scale", "B", "bfloat16", (tables[m]["shape"][0],)))
    plane_specs(m, "B", B_GROUP, B_BITS)
    plane_specs(m, "C", C_GROUP, C_BITS)
for k in unquant:
    shape = tuple(get(ORIG, omap, k).shape)
    dt = str(get(QG4, qmap, k).dtype).split(".")[-1]
    specs.append((f"{k}|N4", "N4", dt, shape))
    specs.append((f"{k}|N8", "N8", str(get(ORIG, omap, k).dtype).split(".")[-1], shape))

meta = {"tables": tables, "widths": {w: int(round(I * f / 64)) * 64 for w, f in WIDTHS.items()},
        "intermediate_size": I, "neuron_order": "results/perm-wanda.npz",
        "planes": {"A": [A_GROUP, A_BITS], "B": [B_GROUP, B_BITS], "C": [C_GROUP, C_BITS]},
        "sources": {"A": QG4, "orig": ORIG}}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
w = NestWriter(OUT, specs, meta)
print(f"planned {len(specs)} tensors, file {w.size / 1e9:.2f} GB", flush=True)

# --- fill ----------------------------------------------------------------------------
t0 = time.time()
for f in EXTRA_FILES:
    w.write(f"file|{f}", open(f"{QG4}/{f}", "rb").read())
w.write("file|config.json", cfg_bytes)
for k in unquant:
    w.write(f"{k}|N4", get(QG4, qmap, k))
    w.write(f"{k}|N8", get(ORIG, omap, k))


def write_plane(m, plane, parts, group, bits, row):
    t = tables[m]
    bl = t["blocks"]
    for b in range(len(bl) - 1):
        tag = f"{plane}{b}" if t["kind"] == "cols" else plane
        c0, c1 = bl[b], bl[b + 1]
        pw, pg = 32 // bits, group
        for part, a, step in zip(("weight", "scales", "biases"), parts, (pw, pg, pg)):
            blk = a if t["kind"] != "cols" else mx.contiguous(a[:, c0 // step: c1 // step])
            w.write(f"{m}|{tag}|{part}", blk, row=row)


errors = {}
cur_layer = None
for m in order:
    t = tables[m]
    if t["layer"] != cur_layer:
        cur_layer = t["layer"]
        _files.clear()
        mx.clear_cache()
        print(f"layer {cur_layer} {time.time() - t0:.0f}s peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
    a_parts = tuple(get(QG4, qmap, f"{m}.{p}") for p in ("weight", "scales", "biases"))
    o = get(ORIG, omap, f"{m}.weight")
    if t["layer"] >= 0 and t["kind"] != "none":
        p = mx.array(perms[f"layer_{t['layer']}"])
        o = o[p] if t["kind"] == "rows" else o[:, p]
    errs = defaultdict(list)
    for r in range(0, t["shape"][0], ROW_CHUNK):
        sl = slice(r, r + ROW_CHUNK)
        pl = make_planes(o[sl], tuple(x[sl] for x in a_parts))
        write_plane(m, "A", tuple(x[sl] for x in a_parts), A_GROUP, A_BITS, r)
        w.write(f"{m}|B|row_scale", pl["row_scale"].reshape(-1).astype(mx.bfloat16), row=r)
        write_plane(m, "B", pl["b_parts"], B_GROUP, B_BITS, r)
        write_plane(m, "C", pl["c_parts"], C_GROUP, C_BITS, r)
        for e in ("err4", "err8", "err16"):
            errs[e].append(pl[e])
        del pl
    errors[m] = {e: float(np.mean(v)) for e, v in errs.items()}
w.close()
json.dump(errors, open("results/build-nest-errors.json", "w"), indent=1)
worst = max(errors.values(), key=lambda e: e["err16"])
print(f"done {time.time() - t0:.0f}s; worst 16-bit table error {worst['err16']:.2e}; "
      f"median 8-bit error {np.median([e['err8'] for e in errors.values()]):.4f}", flush=True)
