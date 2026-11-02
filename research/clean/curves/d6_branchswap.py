"""Where is the write gap? Trained read+write; swap ONE 4a branch's column+bias to the vmr write (and the reverse:
vmr write with ONE branch from training). Dev 1000."""
import sys, numpy as np, torch
from inception import *
from family import Family, Data, evaluate, IN4A
from d4_write import stats, writes
m = load_model(); D = Data(seed=0, ndev=1000); fam = list(np.load("cache/s1_fine.npz")["fam"])
ck = torch.load(f"cache/ckpt/{sys.argv[1]}.pt", map_location="cpu", weights_only=False); fm = Family(m, fam, 1, ks=ck.get("ks", 5)).to(DEV); fm.load_state_dict(ck["state"])
tc = [p.detach().clone() for p in fm.cols]; res, g = writes(m, fam, stats(m, fm, fam, D)); vc, vb = res["vmr"]
vc = [c.to(DEV) / s for c, s in zip(vc, fm.cscale)]; zero = [torch.zeros_like(b) for b in vb]
def setw(cols, cb):
    with torch.no_grad():
        for p, c in zip(fm.cols, cols): p.copy_(c)
    fm.colbias = cb
names = ["1x1", "3x3b", "5x5b", "pool"]
setw(tc, None); print("trained write:", evaluate(m, fm, D, D.dv)[0])
setw(vc, vb); print("vmr write:", evaluate(m, fm, D, D.dv)[0])
for j, n in enumerate(names):
    cols = [vc[i] if i == j else tc[i] for i in range(4)]; cb = [vb[i] if i == j else zero[i] for i in range(4)]
    setw(cols, cb); a = evaluate(m, fm, D, D.dv)[0]
    cols = [tc[i] if i == j else vc[i] for i in range(4)]; cb = [zero[i] if i == j else vb[i] for i in range(4)]
    setw(cols, cb); b = evaluate(m, fm, D, D.dv)[0]
    print(f"branch {n}: trained write but this branch vmr -> {a:.4f} | vmr write but this branch trained -> {b:.4f}", flush=True)
