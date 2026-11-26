"""Union check, step 1 (2026-11-25, plan "reconcile the tree checks with the essay's coarse neuron").
For teacher / kd / baseline: each MLP neuron's MAX firing over all 197 tokens per photo (the strongest crop, the
essay's way of reading a neuron), every block.

Photos for reading only: fit positions 0-7 of each class (8,000). FIT = positions 0-3, SCORE = 4-7.
Held (positions 8-9) and test (odd positions) are not touched. Old files are not touched.
Output: out/maxfire_<model>[_smoke].npz with fire (N, 12, H) fp16, photo_ids, pos (0-7).
Usage: PYTHONPATH=distill/clipkd/pylib curves/.venv/bin/python -I distill/clipkd/read_maxfire.py [--smoke]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_clip import CLIP_MEAN, CLIP_STD, DEV, OUT, net  # noqa: E402
from read_stream import tower  # noqa: E402


def max_fire(v, x):
    fs = []

    def grab(m, i, o):
        if o.shape[0] == 197 and o.shape[1] == x.shape[0]:
            o = o.transpose(0, 1)
        assert o.shape[1] == 197, o.shape
        fs.append(o.max(1).values.float().cpu())

    hs = [b.mlp.gelu.register_forward_hook(grab) for b in v.transformer.resblocks]
    with torch.no_grad():
        v(x)
    for h in hs:
        h.remove()
    return torch.stack(fs, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bs", type=int, default=50)
    a = ap.parse_args()
    t0 = time.time()
    ph = net.Photos("read")
    ids, pos = [], []
    for c in range(1000 if not a.smoke else 25):
        w = np.where(ph.lab == c)[0][0::2]
        ids += list(w[:8]); pos += list(range(8))
    ids, pos = np.array(ids), np.array(pos)
    tag = "_smoke" if a.smoke else ""
    for name in ("teacher", "kd", "baseline"):
        v = tower(name)
        F = []
        for i in range(0, len(ids), a.bs):
            x = torch.from_numpy(ph.raw(ids[i:i + a.bs]).copy()).permute(0, 3, 1, 2).float() / 255.0
            F.append(max_fire(v, ((x - CLIP_MEAN) / CLIP_STD).to(DEV)).half())
        F = torch.cat(F).numpy()
        np.savez(os.path.join(OUT, f"maxfire_{name}{tag}.npz"), fire=F, photo_ids=ids, pos=pos)
        print(name, F.shape, f"{time.time() - t0:.0f}s", flush=True)
        del v, F
        if DEV == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
