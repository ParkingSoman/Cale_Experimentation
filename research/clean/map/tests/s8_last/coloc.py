"""Co-location on the target's top-50 images (20k set): candidate value at the target's peak cell (candidate map pooled
or repeated to the target grid) divided by the candidate's 99.9th pct image max (over these + 450 random images),
and the fraction of images where the candidate's own peak cell is within 1 cell of the target peak. Usage: python coloc.py KEY..."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from win_lib import top_images, maps
import last_lib as L
C = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "s8", "candidates.json")))
X, y, ana = L.data(); rng = np.random.default_rng(0); rnd = rng.choice(ana, 450, replace=False); R = np.stack([np.asarray(X[i]) for i in rnd])
def togrid(z, Rt):
    f = z.shape[1] // Rt
    if f >= 1: B = z.shape[0]; return z.reshape(B, Rt, f, Rt, f).max((2, 4))
    g = Rt // z.shape[1]; return z.repeat(g, 1).repeat(g, 2)
for key in sys.argv[1:]:
    T = tuple(map(int, key.split("_"))); cand = {c: tuple(v) for c, v in C[key].items()}
    imgs, _ = top_images(*T, k=50); Z = maps(np.concatenate([imgs, R]), [T] + list(cand.values()))
    zt = Z[T][:50]; Rt = zt.shape[1]; pk = zt.reshape(50, -1).argmax(1); pr, pc = pk // Rt, pk % Rt
    print(key)
    for c, p in cand.items():
        zc = togrid(Z[p], Rt); ref = np.percentile(zc.max((1, 2)), 99.9); zc = zc[:50]
        at = zc.reshape(50, -1)[np.arange(50), pk] / ref; own = zc.reshape(50, -1).argmax(1)
        near = ((np.abs(own // Rt - pr) <= 1) & (np.abs(own % Rt - pc) <= 1)).mean()
        print(f"  {c} b{p[0]}n{p[1]}: median at-peak {np.median(at):+.2f}  frac>0.25 {(at > .25).mean():.2f}  peak-near {near:.2f}")
