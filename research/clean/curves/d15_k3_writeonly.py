"""Measuring tool: k=3 read frozen at three members, train only the write (columns + 4a bias offsets), start = derived
Bayes write. usage: d15_k3_writeonly.py 330,388,385 [steps]"""
import sys, json, numpy as np, torch, torch.nn as nn
from inception import *
from family import Family, Data, evaluate, train
from d12_k3 import member_stats, build
keep = [int(t) for t in sys.argv[1].split(",")]; steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
m = load_model(); D = Data(seed=0, ndev=1000); fam = list(np.load("cache/s1_fine.npz")["fam"])
for p in m.parameters(): p.requires_grad_(False)
fm = Family(m, fam, len(keep), init="member:" + ",".join(map(str, keep))).to(DEV)
cols, cbs = build(m, fam, keep, member_stats(m, fam, D), "bayes")
with torch.no_grad():
    for p, c, s in zip(fm.cols, cols, fm.cscale): p.copy_(c.to(DEV) / s)
fm.cbp = nn.ParameterList([nn.Parameter(c.clone() / 100.0) for c in cbs])
class FB(list):
    def __getitem__(self, i): return fm.cbp[i] * 100.0
fm.colbias = FB([0, 0, 0, 0]); fm.read.requires_grad_(False); fm.bias.requires_grad_(False)
print("start (derived Bayes):", evaluate(m, fm, D, D.dv), flush=True)
best, step, hist = train(m, fm, D, lr=1e-2, steps=steps, eval_every=250, patience=4)
print(f"BEST k3 write-only {keep}: {best:.4f} at step {step}", flush=True)
torch.save(dict(state=fm.state_dict(), cbp=[p.detach().cpu() for p in fm.cbp], keep=keep), f"cache/ckpt/k3-writeonly-{'-'.join(map(str, keep))}.pt")
json.dump(dict(best=best, step=step, hist=hist), open(f"../../../results/curves-k3-writeonly-{'-'.join(map(str, keep))}.json", "w"), indent=1)
