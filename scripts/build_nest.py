"""Build models/g12-nest/model.nest: the QAT 4-bit 12B, its width tiers and its drafter in one file.

Source: models/qg4 (QAT exact-grid 4-bit, MLP neurons already in importance order)
and models/qat-assistant-4bit (Google's MTP drafter, stored as one raw blob).

MLP widths: gate/up rows are in importance order, so a width is a row prefix.
down_proj is stored as column blocks [0, 85%), [85%, 90%), [90%, 100%), each its own
contiguous matrix, so every width reads contiguous bytes (no strided down_proj read).

(The precision ladder, planes B and C, was dropped on 2026-10-01: the original 16-bit
scored the same as QAT 4-bit. See FINDINGS.md. nest/planes.py keeps that math.)

usage: build_nest.py [OUT] [LAYER]   (default models/g12-nest/model.nest; LAYER: test build of one layer)
"""
import json
import os
import sys
import time

import mlx.core as mx

from nest.nestfile import NestWriter

QG4 = "models/qg4"
DRAFTER = "models/qat-assistant-4bit"
OUT = sys.argv[1] if len(sys.argv) > 1 else "models/g12-nest/model.nest"
ONLY = int(sys.argv[2]) if len(sys.argv) > 2 else None
FILES = ["config.json", "chat_template.jinja", "generation_config.json", "processor_config.json",
         "tokenizer_config.json", "tokenizer.json"]
DRAFTER_FILES = ["config.json", "model.safetensors"]
SKIP = ("embed_audio.",)            # mlx-vlm drops it: the 12B has no audio tower
WIDTHS = {"100": 1.0, "90": 0.90, "85": 0.85}

cfg = json.load(open(f"{QG4}/config.json"))
G, BITS = cfg["quantization"]["group_size"], cfg["quantization"]["bits"]
qmap = json.load(open(f"{QG4}/model.safetensors.index.json"))["weight_map"]
_files = {}


def get(key):
    f = f"{QG4}/{qmap[key]}"
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
I = cfg["text_config"]["intermediate_size"]
cuts = sorted({int(round(I * w / 64)) * 64 for w in WIDTHS.values()})     # 13056, 13824, 15360
tables = {}
for k in qmap:
    if k.endswith(".scales") and not k.startswith(SKIP):
        m = k[: -len(".scales")]
        out_dim, n_groups = get(k).shape
        kind = width_kind(m)
        tables[m] = {"shape": [out_dim, n_groups * G], "kind": kind, "layer": layer_of(m),
                     "blocks": [0] + cuts if kind == "cols" else [0, n_groups * G]}
unquant = sorted(k for k in qmap if not k.startswith(SKIP) and not k.endswith((".scales", ".biases"))
                 and k[: -len(".weight")] not in tables)
if ONLY is not None:
    tables = {m: t for m, t in tables.items() if t["layer"] == ONLY}
    unquant = [k for k in unquant if layer_of(k) == ONLY]
order = sorted(tables, key=lambda m: (tables[m]["layer"], m))

specs = []
for f in FILES:
    specs.append((f"file|{f}", "X", "uint8", (os.path.getsize(f"{QG4}/{f}"),)))
for f in DRAFTER_FILES:
    specs.append((f"file|drafter/{f}", "X", "uint8", (os.path.getsize(f"{DRAFTER}/{f}"),)))
for m in order:
    t = tables[m]
    bl = t["blocks"]
    for b in range(len(bl) - 1):
        cols = bl[b + 1] - bl[b]
        tag = f"A{b}" if t["kind"] == "cols" else "A"
        specs.append((f"{m}|{tag}|weight", "A", "uint32", (t["shape"][0], cols * BITS // 32)))
        specs.append((f"{m}|{tag}|scales", "A", "bfloat16", (t["shape"][0], cols // G)))
        specs.append((f"{m}|{tag}|biases", "A", "bfloat16", (t["shape"][0], cols // G)))
for k in unquant:
    a = get(k)
    specs.append((f"{k}|N4", "N4", str(a.dtype).split(".")[-1], tuple(a.shape)))

meta = {"tables": tables, "widths": {w: int(round(I * f / 64)) * 64 for w, f in WIDTHS.items()},
        "intermediate_size": I, "neuron_order": "results/perm-wanda.npz",
        "planes": {"A": [G, BITS]}, "sources": {"A": QG4, "drafter": DRAFTER}}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
w = NestWriter(OUT, specs, meta)
print(f"planned {len(specs)} tensors, file {w.size / 1e9:.2f} GB", flush=True)

# --- fill ----------------------------------------------------------------------------
t0 = time.time()
for f in FILES:
    w.write(f"file|{f}", open(f"{QG4}/{f}", "rb").read())
for f in DRAFTER_FILES:
    w.write(f"file|drafter/{f}", open(f"{DRAFTER}/{f}", "rb").read())
for k in unquant:
    w.write(f"{k}|N4", get(k))
cur = None
for m in order:
    t = tables[m]
    if t["layer"] != cur:
        cur = t["layer"]
        _files.clear()
        mx.clear_cache()
        if cur % 8 == 0:
            print(f"layer {cur} {time.time() - t0:.0f}s", flush=True)
    parts = [get(f"{m}.{p}") for p in ("weight", "scales", "biases")]
    bl = t["blocks"]
    for b in range(len(bl) - 1):
        tag = f"A{b}" if t["kind"] == "cols" else "A"
        c0, c1 = bl[b], bl[b + 1]
        for p, a, step in zip(("weight", "scales", "biases"), parts, (32 // BITS, G, G)):
            blk = a if t["kind"] != "cols" else mx.contiguous(a[:, c0 // step: c1 // step])
            w.write(f"{m}|{tag}|{p}", blk)
w.close()
print(f"done {time.time() - t0:.0f}s, {w.size / 1e9:.2f} GB", flush=True)
