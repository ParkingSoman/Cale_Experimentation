"""INPUT round 3: make_input's CLIP self-InfoNCE recipe with ONE knob changed: K (steps). Imports make_input read-only.
Object (picked by): the CLIP generic input X (picked by round 2: it moves the stream basis). Photo use: none.
Output: cache/clip_<kind>_n256_k<K>_s1.pt (+ .json).
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_input as mi  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--K", type=int, required=True)
ap.add_argument("--kind", default="self", choices=("self", "null"))
a = ap.parse_args()
torch.manual_seed(0)
path = mi.out_path("clip", a.kind, k=a.K)
assert not os.path.exists(path), path
x_all = torch.load(mi.PINK, weights_only=True)
model, y = mi.load_net("clip", a.kind)
t0 = time.time()
X, log = mi.make("clip", model, y, x_all, k_steps=a.K)
X16 = X.half().contiguous()
assert torch.isfinite(X16).all()
torch.save(X16, path)
meta = dict(net="clip", kind=a.kind, K=a.K, recipe="make_input clip self-InfoNCE, only K changed", photo_use="none",
            minutes=round((time.time() - t0) / 60, 2),
            loss_step1_mean=sum(r["loss_step1"] for r in log) / len(log),
            loss_stepK_mean=sum(r["loss_stepK"] for r in log) / len(log),
            rel_change_mean=sum(r["rel_change"] for r in log) / len(log), batches=log)
json.dump(meta, open(path[:-3] + ".json", "w"), indent=1)
print("saved", path, meta["minutes"], "min; rel change", round(meta["rel_change_mean"], 3))
