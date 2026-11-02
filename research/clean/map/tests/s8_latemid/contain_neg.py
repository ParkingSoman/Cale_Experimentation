"""Negative-pole containment + position bias. For each target: k = number of target on-spots (z > .3 max).
Cneg = P(cand on | target among its k most-negative spots); Cpos same for the k top spots. Also share of the cand's
top-100 image peaks in the border ring (row/col 0 or 13) vs the target's. Usage: python contain_neg.py"""
import os, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); f = .3
D = np.load(os.path.join(H, "out", "scan.npz")); C = json.load(open(os.path.join(H, "..", "..", "s8", "candidates.json")))
def ring(Z):
    v = Z.reshape(len(Z), -1); top = np.argsort(-v.max(1))[:100]; a = v[top].argmax(1); r, c = a // 14, a % 14
    return float(((r == 0) | (r == 13) | (c == 0) | (c == 13)).mean()), float((r >= 11).mean())
for key in ["14_1043", "12_575", "13_404", "12_245", "12_1342"]:
    Zt = D[key].astype(np.float32); T = Zt.ravel(); k = int((T > f * T.max()).sum()); o = np.argsort(T)
    pos, neg = o[-k:], o[:k]; rg = ring(Zt)
    print(f"\n{key}: k={k}  border share {rg[0]:.2f}  bottom-3-rows share {rg[1]:.2f}  (chance border .27, bottom .21)")
    for c, (b, n) in C[key].items():
        Z = D[f"{b}_{n}"].astype(np.float32); z = Z.ravel(); p = z > f * z.max(); rc = ring(Z)
        print(f"  {c} b{b:2d} n{n:<5d} breadth {p.mean():.3f}  Cpos {p[pos].mean():.2f}  Cneg {p[neg].mean():.2f}  border {rc[0]:.2f} bottom {rc[1]:.2f}")
