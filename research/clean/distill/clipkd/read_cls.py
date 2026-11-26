"""Microscope check for Steps 2-3 (2026-11-25): the output reads ONLY the class token after block 12 (pool_type 'tok').
Save, for teacher / kd / baseline, the class-token stream at 13 points and the class-token MLP firing at 12 blocks.
Analysis is in numpy afterwards (analyze_cls.py): same-depth match, cross-depth table, KD-vs-baseline positive control.

Photos for reading only: reading photos, fit positions 0-3 of each class (4,000 photos; position 8-9 = held and the
odd positions = test are not touched). FIT = positions 0-1, SCORE = positions 2-3.
Output: out/cls_<model>.npz with stream (N, 13, D) fp16, fire (N, 12, H) fp16, photo_ids, half (0 fit / 1 score).
Usage: PYTHONPATH=distill/clipkd/pylib curves/.venv/bin/python -I distill/clipkd/read_cls.py [--smoke]
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


def cls_parts(v, x):
    st, fi = [], []
    def grab(store):
        def f(m, i, o):
            if o.shape[0] == 197 and o.shape[1] == x.shape[0]:
                o = o.transpose(0, 1)
            store.append(o[:, 0].float().cpu())
        return f
    hs = [v.ln_pre.register_forward_hook(grab(st))]
    hs += [b.register_forward_hook(grab(st)) for b in v.transformer.resblocks]
    hs += [b.mlp.gelu.register_forward_hook(grab(fi)) for b in v.transformer.resblocks]
    with torch.no_grad():
        v(x)
    for h in hs:
        h.remove()
    return torch.stack(st, 1), torch.stack(fi, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bs", type=int, default=50)
    a = ap.parse_args()
    t0 = time.time()
    ph = net.Photos("read")
    ids, half = [], []
    for c in range(1000 if not a.smoke else 25):
        w = np.where(ph.lab == c)[0][0::2]
        ids += [w[0], w[1], w[2], w[3]]; half += [0, 0, 1, 1]
    ids, half = np.array(ids), np.array(half)
    tag = "_smoke" if a.smoke else ""
    for name in ("teacher", "kd", "baseline"):
        v = tower(name)
        S, Fi = [], []
        for i in range(0, len(ids), a.bs):
            x = torch.from_numpy(ph.raw(ids[i:i + a.bs]).copy()).permute(0, 3, 1, 2).float() / 255.0
            s, f = cls_parts(v, ((x - CLIP_MEAN) / CLIP_STD).to(DEV))
            S.append(s.half()); Fi.append(f.half())
        np.savez(os.path.join(OUT, f"cls_{name}{tag}.npz"), stream=torch.cat(S).numpy(), fire=torch.cat(Fi).numpy(),
                 photo_ids=ids, half=half)
        print(name, torch.cat(S).shape, torch.cat(Fi).shape, f"{time.time() - t0:.0f}s", flush=True)
        del v
        if DEV == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
