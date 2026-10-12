"""Read a .nest file back and check it against its sources.

| Check | Pass |
|---|---|
| every weight table (down_proj blocks re-joined) = models/qg4 | bit-identical |
| every norm / unquantized tensor = models/qg4 | bit-identical |
| drafter blob = models/qat-assistant-4bit files | byte-identical |

usage: check_nest.py NEST_FILE
"""
import json
import sys

import mlx.core as mx

from nest.nestfile import NestFile

nf = NestFile(sys.argv[1])
H = nf.header
src = H["sources"]["A"]
qmap = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
_files = {}


def get(key):
    f = f"{src}/{qmap[key]}"
    if f not in _files:
        _files[f] = mx.load(f)
    return _files[f][key]


bad = []
cur = None
for m, t in sorted(H["tables"].items(), key=lambda e: (e[1]["layer"], e[0])):
    if t["layer"] != cur:
        cur = t["layer"]
        _files.clear()
        mx.clear_cache()
    n = len(t["blocks"]) - 1
    for p in ("weight", "scales", "biases"):
        if t["kind"] == "cols":
            a = mx.concatenate([nf.array(f"{m}|A{b}|{p}") for b in range(n)], axis=1)
        else:
            a = nf.array(f"{m}|A|{p}")
        if not mx.array_equal(a, get(f"{m}.{p}")).item():
            bad.append(f"differs: {m}.{p}")
for name, tt in nf.tensors.items():
    if tt["region"] == "N4" and not mx.array_equal(nf.array(name), get(name.split("|")[0])).item():
        bad.append(f"differs: {name}")
    if name.startswith("file|drafter/"):
        f = name.split("/", 1)[1]
        if nf.blob(name) != open(f"{H['sources']['drafter']}/{f}", "rb").read():
            bad.append(f"differs: {name}")
print(f"tables {len(H['tables'])}, tensors {len(nf.tensors)}, problems {len(bad)}")
for b in bad[:20]:
    print("  ", b)
sys.exit(1 if bad else 0)
