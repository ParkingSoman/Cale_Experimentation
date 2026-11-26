# scale: neurons=A:block0-all, A:block1-weights, A:head-weights; net=A, C-ladder, W-ladder, L-ladder, R4-ladder, R-ladder, WC-ladder, R4C-ladder; photos=saved; where=mac
# rung: Q=plan "Read what training adds on top of the calculation", step 0: ladder builds' stage-4 weights as numpy; N=30 saved .pt files; EST=1 min Mac CPU; PERMIT=Shonu approved plan 2026-11-25
"""Step 0 (Mac): write each ladder build's stage-4 weights in the w_A.npz format (neuron/coarse/weights18.py), so the
reads run in numpy. Output: results/oc_ladder/n<cut>/w_<build>.npz. Also checks that layer3 and earlier equal A's for
every build except R (stages 0-3 frozen there)."""
import glob
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
LAD = os.path.join(HERE, "..", "results", "oc_ladder")
R18 = os.path.join(HERE, "..", "..", "neuron", "results", "coarse", "r18")


def bnrows(sd, p):
    return np.stack([sd[p + ".weight"], sd[p + ".bias"], sd[p + ".running_mean"], sd[p + ".running_var"]])


def export(sd):
    sd = {k: v.numpy() for k, v in sd.items() if v.dtype.is_floating_point}
    out = {}
    for b in (0, 1):
        out[f"W1_{b}"] = sd[f"layer4.{b}.conv1.weight"]
        out[f"W2_{b}"] = sd[f"layer4.{b}.conv2.weight"]
        out[f"bn1_{b}"] = bnrows(sd, f"layer4.{b}.bn1")
        out[f"bn2_{b}"] = bnrows(sd, f"layer4.{b}.bn2")
    out["Wd"] = sd["layer4.0.downsample.0.weight"]
    out["bnd"] = bnrows(sd, "layer4.0.downsample.1")
    out["Wfc"], out["bfc"] = sd["fc.weight"], sd["fc.bias"]
    return out, sd


def main():
    A = np.load(os.path.join(R18, "w_A.npz"))
    ref = None
    for f in sorted(glob.glob(os.path.join(LAD, "n*", "*.pt"))):
        cut, b = os.path.basename(os.path.dirname(f)), os.path.basename(f)[:-3]
        out, sd = export(torch.load(f, map_location="cpu", weights_only=True))
        np.savez(os.path.join(os.path.dirname(f), f"w_{b}.npz"), **out)
        # checks: shapes as A; earlier layers identical across builds except R
        assert all(out[k].shape == A[k].shape for k in A.files), f
        early = {k: v for k, v in sd.items() if not (k.startswith("layer4") or k.startswith("fc"))}
        if ref is None and b != "R":
            ref = early
        same = b == "R" or all(np.array_equal(ref[k], early[k]) for k in ref)
        # untouched stage-4 parts equal A's (C, W, WC, L train block 0 only)
        dif = {k: float(np.abs(out[k] - A[k]).max()) for k in A.files}
        changed = [k for k, v in dif.items() if v > 0]
        print(f"{cut:5s} {b:7s} early layers same: {same}; changed vs A: {changed}")


if __name__ == "__main__":
    main()
