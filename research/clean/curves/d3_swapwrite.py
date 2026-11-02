"""Trained read + Bayes write (columns replaced) vs trained read + trained write. Is the write the gap?"""
import sys, numpy as np, torch
from inception import *
from family import Family, Data, evaluate
from derive import bayes_write
m = load_model(); D = Data(seed=0, ndev=1000); fam = list(np.load("cache/s1_fine.npz")["fam"])
ck = torch.load(f"cache/ckpt/{sys.argv[1]}.pt", map_location="cpu"); fm = Family(m, fam, 1, ks=ck.get("ks", 5)).to(DEV); fm.load_state_dict(ck["state"])
print("trained read + trained write:", evaluate(m, fm, D, D.dv), flush=True)
cols, cbias, beta, alpha = bayes_write(m, fm, fam, D)
tc = [c.clone() for c in fm.weights()[2]]
for nm, c, t in zip(["1x1", "3x3b", "5x5b", "pool"], cols, tc):
    c = c[:, 0].numpy(); t = t[:, 0].cpu().numpy()
    print(f"  {nm}: cos(trained col, Bayes col) {c @ t / np.linalg.norm(c) / np.linalg.norm(t):+.3f}  |trained|/|Bayes| {np.linalg.norm(t)/np.linalg.norm(c):.2f}")
with torch.no_grad():
    for p, c, s in zip(fm.cols, cols, fm.cscale): p.copy_(c.to(DEV) / s)
fm.colbias = cbias
print("trained read + Bayes write:", evaluate(m, fm, D, D.dv), " beta", np.round(beta, 3), flush=True)
