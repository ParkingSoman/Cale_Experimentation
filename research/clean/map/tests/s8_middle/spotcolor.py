"""Colour/texture at a unit's top-100 and bottom-100 spots (32 px patch around the peak cell) vs 300 random spots.
Channels: lum, rg = R-G, by = B-(R+G)/2, sat = max-min, tex = lum std. Usage: python spotcolor.py B,N [...]"""
import sys, os
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", "..", "..", "vision2"))
import gallery as G
T = [tuple(map(int, a.split(","))) for a in sys.argv[1:]]; W = {}
for b, n in T: W.setdefault(b, []).append(n)
G._init(); S = G.scan(W); rng = np.random.default_rng(2)
def stats(ims, cells, R):
    out = []
    for i, p in zip(ims, cells):
        r, c = divmod(int(p), R); s = 224 / R; cy, cx = int((r + .5) * s), int((c + .5) * s)
        a = np.asarray(G._X[G._idx[i]]).astype(np.float32)[max(cy - 16, 0):cy + 16, max(cx - 16, 0):cx + 16]
        R_, G_, B_ = a[..., 0].mean(), a[..., 1].mean(), a[..., 2].mean(); L = a.mean(-1)
        out.append([L.mean(), R_ - G_, B_ - (R_ + G_) / 2, (a.max(-1) - a.min(-1)).mean(), L.std()])
    return np.array(out).mean(0)
ri = rng.choice(len(G._idx), 300); rp = rng.integers(0, 196, 300)
print("unit          pole   lum    rg     by     sat    tex")
print("random            " + " ".join(f"{v:6.1f}" for v in stats(ri, rp, 14)))
for t in T:
    vm, pm, vn, pn = S[t]; R = 56 // 2 ** G._m.stage_of[t[0]]
    for pole, o, p in (("pos", np.argsort(-vm)[:100], pm), ("neg", np.argsort(vn)[:100], pn)):
        print(f"b{t[0]}n{t[1]:<8d} {pole}  " + " ".join(f"{v:6.1f}" for v in stats(o, p[o], R)))
