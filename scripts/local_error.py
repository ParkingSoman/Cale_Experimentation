"""Layer-local test on real hidden states (bf16, layer by layer).

For selected layers and widths, measure the relative error of the MLP output
when only the top-n neurons are kept:
  - ranking: wanda (rms x weight norm) vs flap (variance x weight norm^2)
  - repair:  none vs mean compensation (add back the removed neurons' average output)
Uses held-out samples, so the statistics (from calibration data) are not tested on themselves.
"""
import sys

import mlx.core as mx
import numpy as np
from mlx_vlm.models.gemma4 import language as g4

sys.path.insert(0, ".")
from nest.io import read_jsonl
from nest.levels import _moments, load_stats, scores
from nest.stream import LayerwiseRunner

LAYERS = [2, 8, 11, 16, 24, 32, 40, 46]
WIDTHS = [13824, 11520, 7680]
st = load_stats()
e1, _, _, _ = _moments(st)
orders = {k: np.argsort(-scores(st, k), axis=1) for k in ("wanda", "flap")}
results = {}
CUR = {"i": None}


def mlp_call(self, x):
    a = g4.geglu(self.gate_proj(x), self.up_proj(x))
    i = CUR["i"]
    if i in LAYERS:
        full = self.down_proj(a).astype(mx.float32)
        W = self.down_proj.weight  # (hidden, inter)
        for kind, order in orders.items():
            for n in WIDTHS:
                keep = mx.array(order[i][:n])
                drop = order[i][n:]
                part = (a[..., keep] @ W[:, keep].T).astype(mx.float32)
                comp = mx.array((np.array(W[:, mx.array(drop)].astype(mx.float32)) @ e1[i][drop]).astype(np.float32))
                for rep, y in (("none", part), ("mean", part + comp)):
                    err = (mx.linalg.norm(y - full) / mx.linalg.norm(full)).item()
                    results.setdefault((i, kind, n, rep), []).append(err)
    return self.down_proj(a)


g4.MLP.__call__ = mlp_call
recs = [r for r in read_jsonl("data/heldout.jsonl") if int(r["id"].split("-")[-1]) % 8 == 0]
runner = LayerwiseRunner("models/gemma-4-12B-it-bf16")
runner.prepare(recs)
Cls = runner.lm.layers[0].__class__
_call = Cls.__call__


def layer_call(self, *a, **k):
    CUR["i"] = self.layer_idx
    return _call(self, *a, **k)


Cls.__call__ = layer_call
runner.run()
print(f"{len(recs)} held-out samples. Relative MLP output error (lower is better):")
print("layer  width  | wanda/none wanda/mean | flap/none flap/mean")
for i in LAYERS:
    for n in WIDTHS:
        r = {(k, rep): np.mean(results[(i, k, n, rep)]) for k in orders for rep in ("none", "mean")}
        print(f"{i:5d} {n:6d}  |   {r[('wanda','none')]:.3f}     {r[('wanda','mean')]:.3f}    |   {r[('flap','none')]:.3f}     {r[('flap','mean')]:.3f}")
