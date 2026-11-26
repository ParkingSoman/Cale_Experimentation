"""Tree check, step 3 (2026-11-25, plan pasted-content-id-0373-while-federated-coral): save the class-token MLP INPUT
(input of mlp.c_fc = ln_2 of the mid-block stream) for teacher / kd / baseline, plus the MLP weights.
Pre-activation of neuron i = W_fc[i] . x + b_fc[i]. Same photos as read_cls.py (4,000 fit-position photos).

Photos for reading only. Old files are not touched. Output: out/cfc_<model>[_smoke].npz with
x (N, 12, D) fp16, W_fc (12, H, D), b_fc (12, H), W_proj (12, D, H), ln2_w (12, D), photo_ids, half.
Usage: PYTHONPATH=distill/clipkd/pylib curves/.venv/bin/python -I distill/clipkd/read_cfc.py [--smoke]
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


def cfc_inputs(v, x):
    xs = []

    def grab(m, i):
        o = i[0]
        if o.shape[0] == 197 and o.shape[1] == x.shape[0]:
            o = o.transpose(0, 1)
        assert o.shape[1] == 197, o.shape
        xs.append(o[:, 0].float().cpu())

    hs = [b.mlp.c_fc.register_forward_pre_hook(grab) for b in v.transformer.resblocks]
    with torch.no_grad():
        v(x)
    for h in hs:
        h.remove()
    return torch.stack(xs, 1)


def weights(v):
    bl = v.transformer.resblocks
    f = lambda t: t.detach().float().cpu().numpy()
    return dict(W_fc=np.stack([f(b.mlp.c_fc.weight) for b in bl]), b_fc=np.stack([f(b.mlp.c_fc.bias) for b in bl]),
                W_proj=np.stack([f(b.mlp.c_proj.weight) for b in bl]), ln2_w=np.stack([f(b.ln_2.weight) for b in bl]))


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
        X = []
        for i in range(0, len(ids), a.bs):
            x = torch.from_numpy(ph.raw(ids[i:i + a.bs]).copy()).permute(0, 3, 1, 2).float() / 255.0
            X.append(cfc_inputs(v, ((x - CLIP_MEAN) / CLIP_STD).to(DEV)).half())
        X = torch.cat(X).numpy()
        np.savez(os.path.join(OUT, f"cfc_{name}{tag}.npz"), x=X, photo_ids=ids, half=half, **weights(v))
        print(name, X.shape, f"{time.time() - t0:.0f}s", flush=True)
        del v
        if DEV == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
