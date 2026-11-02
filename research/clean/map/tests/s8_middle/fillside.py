"""Does a flat fill on one side create a response at the fill/content boundary? 60 random images (not top images).
Variants: original, and a 48 px strip on the left/right/top/bottom replaced by flat mean-gray / black / white
(content is shifted, not covered). Prints mean image max and mean 14-col / 14-row profile of the map.
Usage: python fillside.py B,N [...]"""
import sys, os
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", "..", "..", "vision2")); sys.path.insert(0, os.path.join(H, "..", "s6"))
import gallery as G
from judge_lib import maps
T = [tuple(map(int, a.split(","))) for a in sys.argv[1:]]
G._init(); sel = np.random.default_rng(5).choice(len(G._idx), 60, replace=False)
nat = np.stack([np.asarray(G._X[G._idx[i]]) for i in sel])
FILL = {"mean": (124, 116, 104), "black": (0, 0, 0), "white": (255, 255, 255)}
def var(side, fill):
    out = nat.copy(); D = 48
    if side == "left": out[:, :, D:] = nat[:, :, :-D]; out[:, :, :D] = fill
    if side == "right": out[:, :, :-D] = nat[:, :, D:]; out[:, :, -D:] = fill
    if side == "top": out[:, D:] = nat[:, :-D]; out[:, :D] = fill
    if side == "bottom": out[:, :-D] = nat[:, D:]; out[:, -D:] = fill
    return out
V = {"orig": nat}
for s in ("left", "right", "top", "bottom"):
    for f in ("mean", "black", "white"): V[f"{s}_{f}"] = var(s, FILL[f])
np.set_printoptions(linewidth=200, precision=2, suppress=True)
R = {k: maps(v, T) for k, v in V.items()}
for t in T:
    print(f"\n=== b{t[0]} n{t[1]}")
    for k in V:
        z = R[k][t]; print(f"{k:14s} imgmax {z.max((1,2)).mean():+.2f}  cols {z.mean((0,1))}  rows {z.mean((0,2))}")
