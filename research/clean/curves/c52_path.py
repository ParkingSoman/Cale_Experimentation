"""Fast check (Part VIII, C1 path): does the exact-uniform PCG path head toward training's move T - S, then overshoot?"""
import numpy as np, torch, time
SRC = open("c21_newton.py").read(); exec(SRC.split("vS = Vs[\"S\"] * mask")[0])
S, T = Vs["S"] * mask, Vs["T"] * mask; dT = (T - S) * free
kf = lambda a, b: float((a * (Gf @ (b * free) @ Af)).sum())          # cost metric G (x) A for parameter moves
print(f"|T - S| plain {float(dT.norm()):.4g}")
print(f"{'it':>3s} {'|V-S|/|T-S|':>12s} {'cos plain':>10s} {'cos KFC':>9s} {'proj on T-S':>12s} {'resid/|T-S|':>12s}")
for k in range(1, 16):
    d = (torch.load(f"cache/c50_photo2000_it{k}_V.pt").to(DEV) - S) * free
    c = float((d * dT).sum() / d.norm() / dT.norm()); ck = kf(d, dT) / (kf(d, d) * kf(dT, dT)) ** .5
    a = float((d * dT).sum() / (dT * dT).sum()); res = float((d - a * dT).norm() / dT.norm())
    print(f"{k:3d} {float(d.norm() / dT.norm()):12.3f} {c:10.3f} {ck:9.3f} {a:12.3f} {res:12.3f}", flush=True)
