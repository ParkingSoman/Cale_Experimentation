"""Diagnostic: feed an exact quantity built from the fine family (not a neuron) to the readers, with the Bayes write.
H_r = moved ReLU sum (r = 0 is the plain family sum at q). Which quantity do the readers need?"""
import sys, numpy as np, torch
from inception import *
from family import Family, Data, evaluate
from derive import bayes_write
from derive2 import moved_sum_t
m = load_model(); D = Data(seed=0, ndev=1000); fam = list(np.load("cache/s1_fine.npz")["fam"])
class Oracle(Family):
    def __init__(self, r): super().__init__(m, fam, 1); self.r = r
    def pre(self, ins): return moved_sum_t(F.relu(pre_3b(m, ins[0])[:, fam]), self.r)[:, None]
for r in [float(t) for t in sys.argv[1:]]:
    fm = Oracle(r).to(DEV); cols, cbias, beta, alpha = bayes_write(m, fm, fam, D)
    with torch.no_grad():
        for p, c, s in zip(fm.cols, cols, fm.cscale): p.copy_(c.to(DEV) / s)
    fm.colbias = cbias
    print(f"oracle H_r r={r}: dev KL {evaluate(m, fm, D, D.dv)}  beta {np.round(beta, 3)}", flush=True)
