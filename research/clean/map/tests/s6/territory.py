"""Image-level territory containment (20k analysis images; cache has blocks 11-17). on = image max > .5 x the
neuron's 99.9th-pct max; neg = image min < .5 x 0.1th-pct min. C = P(cand on | targ on), back = P(targ on | cand on),
off = P(cand neg | targ on). Usage: python territory.py 15_1613"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from last_lib import amax, amin
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
key = sys.argv[1]; C = json.load(open(os.path.join(D, "s6", "candidates.json")))[key]; tb, tn = map(int, key.split("_"))
on = lambda b, n, f=.5: amax(b, n) > f * np.percentile(amax(b, n), 99.9)
neg = lambda b, n: amin(b, n) < .5 * np.percentile(amin(b, n), .1)
for f in (.5, .25):
    t = on(tb, tn, f); print(f"\nthreshold {f}: target breadth {t.mean():.4f}")
    for c, (b, n) in C.items():
        if b < 11: print(f"  {c} b{b} n{n}: not cached"); continue
        p = on(b, n, f); q = neg(b, n); cc = np.corrcoef(amax(b, n), amax(tb, tn))[0, 1]
        print(f"  {c} b{b:2d} n{n:<5d} breadth {p.mean():.4f}  C {p[t].mean():.2f}  lift {p[t].mean()/max(p.mean(),1e-9):5.1f}  back {t[p].mean():.2f}  off {q[t].mean():.2f} (negbr {q.mean():.3f})  corr(imgmax) {cc:+.2f}")
