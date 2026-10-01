"""Read a .nest file back and check it against its sources.

| Check | Pass |
|---|---|
| plane A = models/qg4 (down_proj blocks re-joined) | bit-identical |
| N4 = qg4 tensors, N8 = original tensors | bit-identical |
| row_scale * A + B + C vs original (neuron order applied) | relative error < 1e-3 |

usage: check_nest.py NEST_FILE
"""
import json
import sys

import mlx.core as mx
import numpy as np

from nest.nestfile import NestFile
from nest.planes import deq, rel

nf = NestFile(sys.argv[1])
H = nf.header
src_a, src_o = H["sources"]["A"], H["sources"]["orig"]
qmap = json.load(open(f"{src_a}/model.safetensors.index.json"))["weight_map"]
omap = json.load(open(f"{src_o}/model.safetensors.index.json"))["weight_map"]
perms = np.load(H["neuron_order"])
_files = {}


def get(src, wm, key):
    f = f"{src}/{wm[key]}"
    if f not in _files:
        _files[f] = mx.load(f)
    return _files[f][key]


def parts(m, plane, t):
    """Plane parts of table m, with down_proj column blocks joined back into one matrix."""
    if t["kind"] != "cols":
        return tuple(nf.array(f"{m}|{plane}|{p}") for p in ("weight", "scales", "biases"))
    n = len(t["blocks"]) - 1
    return tuple(mx.concatenate([nf.array(f"{m}|{plane}{b}|{p}") for b in range(n)], axis=1)
                 for p in ("weight", "scales", "biases"))


bad, worst = [], 0.0
cur = None
for m, t in sorted(H["tables"].items(), key=lambda e: (e[1]["layer"], e[0])):
    if t["layer"] != cur:
        cur = t["layer"]
        _files.clear()
        mx.clear_cache()
    a = parts(m, "A", t)
    for x, p in zip(a, ("weight", "scales", "biases")):
        if not mx.array_equal(x, get(src_a, qmap, f"{m}.{p}")).item():
            bad.append(f"A differs: {m}.{p}")
    G, Bits = H["planes"]["A"]
    o = get(src_o, omap, f"{m}.weight")
    if t["layer"] >= 0 and t["kind"] != "none":
        p = mx.array(perms[f"layer_{t['layer']}"])
        o = o[p] if t["kind"] == "rows" else o[:, p]
    rs = nf.array(f"{m}|B|row_scale").astype(mx.float32)[:, None]
    step = 16384
    for r in range(0, t["shape"][0], step):
        sl = slice(r, r + step)
        rec = rs[sl] * deq(tuple(x[sl] for x in a), G, Bits)
        rec = rec + deq(tuple(x[sl] for x in parts(m, "B", t)), *H["planes"]["B"])
        rec = rec + deq(tuple(x[sl] for x in parts(m, "C", t)), *H["planes"]["C"])
        e = rel(o[sl].astype(mx.float32) - rec, o[sl].astype(mx.float32))
        worst = max(worst, e)
        if e >= 1e-3:
            bad.append(f"16-bit error {e:.2e}: {m} rows {r}")
for name, tt in nf.tensors.items():
    if tt["region"] in ("N4", "N8"):
        k = name.split("|")[0]
        ref = get(src_a, qmap, k) if tt["region"] == "N4" else get(src_o, omap, k)
        if not mx.array_equal(nf.array(name), ref).item():
            bad.append(f"{tt['region']} differs: {k}")
print(f"tables {len(H['tables'])}, worst 16-bit error {worst:.2e}, problems {len(bad)}")
for b in bad[:20]:
    print("  ", b)
sys.exit(1 if bad else 0)
