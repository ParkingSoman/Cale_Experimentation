"""b15 n1613 P5a (mask below / above the peak on top-20 images) and P5b (class means), plus candidates."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from judge_lib import maps
from last_lib import amax, class_mean, names, data
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
C = json.load(open(os.path.join(D, "s6", "candidates.json")))["15_1613"]; T = (15, 1613)
X, y, ana = data(); a = amax(*T); top = ana[np.argsort(-a)[:20]]
imgs = np.stack([np.asarray(X[i]) for i in top]); z0 = maps(imgs, [T])[T]
pk = [np.unravel_index(z.argmax(), z.shape) for z in z0]; base = z0.reshape(20, -1).max(1)
below, above = imgs.copy(), imgs.copy()
for j, (r, c) in enumerate(pk):
    g = imgs[j].reshape(-1, 3).mean(0).astype(np.uint8)
    below[j, min(224, (r + 1) * 32):] = g; above[j, :r * 32] = g
for nm, arr in (("mask below peak", below), ("mask above peak", above)):
    z = maps(arr, [T])[T].reshape(20, -1).max(1); print(f"P5a {nm}: median ratio {np.median(z / base):.2f}")
nmz = names(); cm = {t: class_mean(amax(*t)) for t in [T] + [tuple(v) for v in C.values()]}
W = ["lakeside", "boathouse", "dock", "canoe", "sandbar", "seashore"]; L = ["alp", "valley", "volcano", "cliff dwelling"]
idx = lambda L: [nmz.index(n) for n in L]
print("P5b class means (image max). columns: target then candidates")
for grp, ls in (("water-edge", W), ("dry landscape", L)):
    print(f"  {grp:14s}", " ".join(f"{np.mean(cm[t][idx(ls)]):6.2f}" for t in cm))
print("  ratio         ", " ".join(f"{np.mean(cm[t][idx(W)]) / max(np.mean(cm[t][idx(L)]), 1e-6):6.2f}" for t in cm))
o = np.argsort(-cm[T])[:15]; print("target top classes:", [(nmz[i], round(float(cm[T][i]), 2)) for i in o])
for c, v in C.items():
    o2 = np.argsort(-cm[tuple(v)])[:8]; print(f"{c} {v} top classes:", [nmz[i] for i in o2])
