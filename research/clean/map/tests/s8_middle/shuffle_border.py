"""Shuffle edits with the frame in mind (4x4 grid of 56 px tiles, 5 random shuffles each).
full   = all 16 tiles shuffled;  inner = only the 4 inner tiles shuffled (12 border tiles kept in place);
border = only the 12 border tiles shuffled among themselves (inner kept).
For the top-30 (pos) and bottom-30 (neg) images: median ratio of image max (pos) or image min (neg) vs original, and
'same cell' = share of shuffled images whose peak (or trough) stays in the original cell (+-1).
Usage: python shuffle_border.py B,N [...]"""
import sys, os, json
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", "..", "..", "vision2")); sys.path.insert(0, os.path.join(H, "..", "s6"))
import gallery as G
from judge_lib import maps
T = [tuple(map(int, a.split(","))) for a in sys.argv[1:]]; W = {}
for b, n in T: W.setdefault(b, []).append(n)
G._init(); S = G.scan(W); rng = np.random.default_rng(0)
INNER = [5, 6, 9, 10]; BORDER = [i for i in range(16) if i not in INNER]
def shuf(im, which):
    t = [im[r * 56:(r + 1) * 56, c * 56:(c + 1) * 56] for r in range(4) for c in range(4)]
    perm = list(range(16)); sub = list(which); p = rng.permutation(sub)
    for a, b in zip(sub, p): perm[a] = b
    out = np.empty_like(im)
    for k in range(16): r, c = divmod(k, 4); out[r * 56:(r + 1) * 56, c * 56:(c + 1) * 56] = t[perm[k]]
    return out
for t in T:
    vm, pm, vn, pn = S[t]; res = {}
    for pole, order in (("pos", np.argsort(-vm)[:30]), ("neg", np.argsort(vn)[:30])):
        imgs = np.stack([np.asarray(G._X[G._idx[i]]) for i in order]); z0 = maps(imgs, [t])[t]
        f = (lambda z: z.max((1, 2))) if pole == "pos" else (lambda z: z.min((1, 2)))
        a = (lambda z: np.argmax(z)) if pole == "pos" else (lambda z: np.argmin(z))
        for name, which in (("full", range(16)), ("inner", INNER), ("border", BORDER)):
            rs, same = [], []
            for rep in range(5):
                z1 = maps(np.stack([shuf(im, which) for im in imgs]), [t])[t]
                rs.append(f(z1) / f(z0))
                for u, v in zip(z0, z1):
                    p0 = np.array(divmod(int(a(u)), 14)); p1 = np.array(divmod(int(a(v)), 14)); same.append(np.abs(p0 - p1).max() <= 1)
            res[f"{pole}_{name}"] = (round(float(np.median(np.concatenate(rs))), 2), round(float(np.mean(same)), 2))
    print(f"b{t[0]} n{t[1]}", json.dumps(res))
