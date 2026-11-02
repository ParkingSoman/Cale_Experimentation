"""Ablation for the derived neuron: intercept only (coarse columns zero, 4a biases get the Bayes intercept = member means)."""
import numpy as np, torch
from inception import *
from family import Family, Data, evaluate
from derive import derived_read, bayes_write
m = load_model(); D = Data(seed=0, ndev=1000); fam = list(np.load("cache/s1_fine.npz")["fam"])
fm = Family(m, fam, 1, ks=13).to(DEV); K, b = derived_read(m, fam, 4)
_, _, c0 = fm.weights(); fm.set_weights(torch.tensor(K)[None].to(DEV), torch.tensor([b]).to(DEV), [c.to(DEV) for c in c0])
cols, cbias, beta, alpha = bayes_write(m, fm, fam, D)
fm.set_weights(torch.tensor(K)[None].to(DEV), torch.tensor([b]).to(DEV), [torch.zeros_like(c).to(DEV) for c in cols]); fm.colbias = cbias
print("intercept only (family -> its Bayes intercept):", evaluate(m, fm, D, D.dv))
