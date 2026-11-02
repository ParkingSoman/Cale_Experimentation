"""Blind lineup numbers for one target vs its 9 candidates (3000-image set).
For each candidate (maps pooled to the target's 14x14 grid; max-pool for 'on', mean-pool for correlation):
 top  = mean cand activation / cand max (99.9 pct of image maxima) at the target's top-60 spots
 neg  = same at the target's 60 most negative spots;  base = same over all spots of 60 random images
 on|T = P(cand > .3 max at spot | target > .5 max)   lift = on|T / P(cand > .3 max) over random spots
 r    = spatial correlation of maps on target's top-60 images (all positions pooled)
 edge = share of cand's top-150 image peaks on the frame border (null .27 at 14x14)
Usage: python lineup.py KEY   (KEY like 10_95)"""
import sys, os, json
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", "..", "..", "vision2")); sys.path.insert(0, os.path.join(H, "..", "s6"))
import gallery as G
from judge_lib import maps
key = sys.argv[1]; T = tuple(map(int, key.split("_")))
C = json.load(open(os.path.join(H, "..", "..", "s8", "candidates.json")))[key]
tg = [T] + [tuple(v) for v in C.values()]; W = {}
for b, n in tg: W.setdefault(b, []).append(n)
G._init(); S = G.scan(W)
def pool(z, R, how):
    k = z.shape[1] // R
    if k == 1: return z
    z = z.reshape(z.shape[0], R, k, R, k); return z.max((2, 4)) if how == "max" else z.mean((2, 4))
vm, pm, vn, pn = S[T]; top = np.argsort(-vm)[:60]; bot = np.argsort(vn)[:60]
rnd = np.random.default_rng(1).choice(len(G._idx), 60, replace=False)
sel = np.concatenate([top, bot, rnd]); imgs = np.stack([np.asarray(G._X[G._idx[i]]) for i in sel])
Z = maps(imgs, tg); R = 14; zt = Z[T]; tmax = np.percentile(vm, 99.9)
print(f"target b{T[0]} n{T[1]} max {tmax:.2f}")
ton = zt[120:] > .5 * tmax; ton_all = zt[:120] > .5 * tmax
for c, (b, n) in C.items():
    cm = max(np.percentile(S[(b, n)][0], 99.9), 1e-6)
    zx = pool(Z[(b, n)], R, "max") / cm; zm = pool(Z[(b, n)], R, "mean")
    tv = np.array([zx[j][divmod(int(np.argmax(zt[j])), R)] for j in range(60)])
    nv = np.array([zx[60 + j][divmod(int(np.argmin(zt[60 + j])), R)] for j in range(60)])
    base = zx[120:].mean(); onT = (zx[:120][ton_all] > .3).mean(); br = (zx[120:] > .3).mean()
    tq = zt[120:] > .25 * tmax; cq = zx[120:] > .3; back = (tq & cq).sum() / max(cq.sum(), 1); tbr = tq.mean()
    r = np.corrcoef(zt[:60].ravel(), zm[:60].ravel())[0, 1]
    rr = 56 // 2 ** G._m.stage_of[b]; p = S[(b, n)][1][np.argsort(-S[(b, n)][0])[:150]]
    pr, pc = p // rr, p % rr; edge = ((pr == 0) | (pr == rr - 1) | (pc == 0) | (pc == rr - 1)).mean()
    rowm, colm = pr.mean() / (rr - 1), pc.mean() / (rr - 1)
    print(f"{c} b{b:2d} n{n:<5d} top {tv.mean():+.2f}  neg {nv.mean():+.2f}  base {base:+.2f}  on|T {onT:.2f} lift {onT/max(br,1e-3):4.1f}  br {br:.3f} (tgt.25 {tbr:.3f}) back {back:.2f}  r {r:+.2f}  edge {edge:.2f} rowpos {rowm:.2f} colpos {colm:.2f}")
p = pm[np.argsort(-vm)[:150]]; print(f"target edge {((p//R==0)|(p//R==R-1)|(p%R==0)|(p%R==R-1)).mean():.2f} rowpos {(p//R).mean()/13:.2f} colpos {(p%R).mean()/13:.2f}")
