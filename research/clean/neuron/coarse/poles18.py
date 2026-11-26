"""N11 reading helper: class profiles of ResNet-18 stage-4 neurons from read_<tag>.npz.
pos profile  = per-class mean of pos (max pre-ReLU input over positions), z-scored over the 1,000 class means.
neg profile  = per-class mean of mean (mean pre-ReLU input), z-scored; the lowest classes = the negative pole.
car score    = mean z of pos over the 11 car classes.
"""
import json, os, sys
import numpy as np
import r18
NAMES = r18.net.class_names()

def profiles(tag):
    d = np.load(os.path.join(r18.OUT, f"read_{tag}.npz"))
    lab = d["lab"]; out = {}
    cnt = np.bincount(lab, minlength=1000)
    for b in (0, 1):
        for kind in ("pos", "mean"):
            v = d[f"{kind}{b}"].astype(np.float32)
            cm = np.zeros((1000, v.shape[1]), np.float32); np.add.at(cm, lab, v); cm /= cnt[:, None]
            out[(kind, b)] = (cm - cm.mean(0)) / cm.std(0).clip(1e-6)
    return out, d

def card(P, b, j, n=6):
    pz, mz = P[("pos", b)][:, j], P[("mean", b)][:, j]
    top = np.argsort(-pz)[:n]; bot = np.argsort(mz)[:n]
    return ("+ " + ", ".join(f"{NAMES[c]} {pz[c]:.1f}" for c in top), "- " + ", ".join(f"{NAMES[c]} {mz[c]:.1f}" for c in bot))

if __name__ == "__main__":
    tag = sys.argv[1]; P, d = profiles(tag)
    for b in (1, 0):
        car = P[("pos", b)][r18.CARS].mean(0)
        order = np.argsort(-car)[:int(sys.argv[2]) if len(sys.argv) > 2 else 20]
        print(f"== {tag} stage-4 block {b}: neurons by car score")
        for j in order:
            p, m = card(P, b, j)
            print(f"b{b} n{j:3d} car {car[j]:.2f} | {p}\n            {m}")
