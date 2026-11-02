"""Measuring tool: freeze a read (derived), train only the write (4a columns + 4a bias offsets), start = vmr write.
usage: d5_writeonly.py <derived2 tag> [steps]"""
import sys, json, numpy as np, torch, torch.nn as nn
from inception import *
from family import Family, Data, evaluate, train
from d4_write import stats, writes
tag = sys.argv[1]; steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
m = load_model(); D = Data(seed=0, ndev=1000); fam = list(np.load("cache/s1_fine.npz")["fam"])
for p in m.parameters(): p.requires_grad_(False)
ck = torch.load(f"cache/ckpt/{tag}.pt", map_location="cpu", weights_only=False); fm = Family(m, fam, 1, ks=ck.get("ks", 5)).to(DEV)
fm.load_state_dict(ck["state"])
res, g = writes(m, fam, stats(m, fm, fam, D)); cols, cb = res["vmr"]
with torch.no_grad():
    for p, c, s in zip(fm.cols, cols, fm.cscale): p.copy_(c.to(DEV) / s)
fm.cbp = nn.ParameterList([nn.Parameter(c.clone() / 100.0) for c in cb])   # bias offsets in units of 100
class FB(list):
    def __getitem__(self, i): return fm.cbp[i] * 100.0
fm.colbias = FB([0, 0, 0, 0])
fm.read.requires_grad_(False); fm.bias.requires_grad_(False)
print("start (vmr):", evaluate(m, fm, D, D.dv), flush=True)
best, step, hist = train(m, fm, D, lr=1e-2, steps=steps, eval_every=250, patience=4)
print(f"BEST write-only {tag}: {best:.4f} at step {step}", flush=True)
json.dump(dict(tag=tag, best=best, step=step, hist=hist), open(f"../../../results/curves-writeonly-{tag}.json", "w"), indent=1)
