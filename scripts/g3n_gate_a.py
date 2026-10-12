"""Gate A, part 1: is the official 4-bit E2B an exact slice of the 4-bit E4B?

Run: .venv/bin/python scripts/g3n_gate_a.py E4B_DIR E2B_DIR   (writes results/g3n-gate-a.json)
Each E2B tensor is compared with the matching piece of the E4B tensor:
  layers 0-19 and 25-34 of E4B become layers 0-19 and 20-29 of E2B,
  gate/up keep the first 8192 rows, down keeps the first 8192 columns,
  the per-layer embedding tables keep the columns (or rows) of the kept layers.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import mlx.core as mx

SKIP = set(range(20, 25))
KEPT = [i for i in range(35) if i not in SKIP]
PLE = 256                                   # hidden_size_per_layer_input


def load_all(d):
    w = {}
    for f in sorted(Path(d).glob("*.safetensors")):
        w.update(mx.load(str(f)))
    return {k: v for k, v in w.items() if ".language_model." in k or k.startswith("language_model.")}


def e4b_name(name):
    m = re.search(r"\.layers\.(\d+)\.", name)
    if not m:
        return name, None
    j = int(m.group(1))
    return name.replace(f".layers.{j}.", f".layers.{KEPT[j]}."), KEPT[j]


def expected(name, t4, t2):
    """The piece of the E4B tensor that the E2B tensor should equal."""
    if "mlp.gate_proj" in name or "mlp.up_proj" in name:
        return t4[: t2.shape[0]]
    if "mlp.down_proj" in name:
        return t4[:, : t2.shape[1]]
    if "embed_tokens_per_layer" in name:
        per = t4.shape[1] // 35               # columns per layer (packed or grouped, same ratio)
        return mx.concatenate([t4[:, i * per:(i + 1) * per] for i in KEPT], axis=1)
    if "per_layer_model_projection" in name:
        if t4.ndim == 2 and t4.shape[0] == 35 * PLE:
            return mx.concatenate([t4[i * PLE:(i + 1) * PLE] for i in KEPT], axis=0)
        # quantized: scales/biases/weight rows follow the same row layout
        return mx.concatenate([t4[i * PLE:(i + 1) * PLE] for i in KEPT], axis=0)
    return t4


def main(d4, d2):
    w4, w2 = load_all(d4), load_all(d2)
    kinds, bad = Counter(), []
    for name, t2 in w2.items():
        n4, _ = e4b_name(name)
        if n4 not in w4:
            kinds["missing in E4B"] += 1
            bad.append((name, "missing"))
            continue
        want = expected(name, w4[n4], t2)
        if want.shape != t2.shape:
            kinds["shape mismatch"] += 1
            bad.append((name, f"{tuple(want.shape)} vs {tuple(t2.shape)}"))
            continue
        same = bool(mx.array_equal(want, t2).item())
        if same:
            kinds["identical"] += 1
        else:
            diff = float(mx.max(mx.abs(want.astype(mx.float32) - t2.astype(mx.float32))).item())
            kinds["different"] += 1
            bad.append((name, f"max abs diff {diff:.4g}"))
    print(dict(kinds))
    for n, why in bad[:25]:
        print("  ", n, why)
    Path("results/g3n-gate-a.json").write_text(json.dumps({"counts": dict(kinds), "first_problems": bad[:50]}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
