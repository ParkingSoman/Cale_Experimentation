"""Trained ceiling for the k=1024 form: start from the no-training build (top-value reads, re-solved writes), then
train with the same loss as distill.py. If training gains little, the build is near the best of this form."""
import json, os, sys
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
EP = int(sys.argv[2]) if len(sys.argv) > 2 else 30
import numpy as np
import torch
import torch.nn.functional as F
from cache import DEV, load
from distill import D, MLP

torch.set_grad_enabled(True)
t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
U = torch.from_numpy(np.load(os.path.join(D, "U_train.npy")))
Y = torch.from_numpy(np.load(os.path.join(D, "Y_train.npy")))
Ut = t(np.load(os.path.join(D, "U_test.npy"))).to(DEV)
Yt = t(np.load(os.path.join(D, "Y_test.npy"))).to(DEV)
m = load("base"); blk = m.features[7][0]
with torch.no_grad():
    W1 = blk.block[3].weight.detach().float().cpu().clone(); b1 = blk.block[3].bias.detach().float().cpu().clone()
    ls = blk.layer_scale.detach().float().cpu().view(-1)
    Wb = ls[:, None] * blk.block[5].weight.detach().float().cpu(); bb = ls * blk.block[5].bias.detach().float().cpu()
    Us = U[::6].float()
    value = F.gelu(Us @ W1.T + b1).std(0) * Wb.norm(dim=0)
    sel = torch.argsort(-value)[:1024]
    g = F.gelu(Us @ W1[sel].T + b1[sel]); X = torch.cat([g, torch.ones(len(g), 1)], 1).double()
    A = X.T @ X; A += 1e-3 * A.diagonal().mean() * torch.eye(len(A), dtype=torch.float64)
    Wn = torch.linalg.solve(A, X.T @ Y[::6].double()).float()
del m
net = MLP(1024)
with torch.no_grad():
    net.l1.weight.copy_(W1[sel]); net.l1.bias.copy_(b1[sel]); net.l2.weight.copy_(Wn[:1024].T); net.l2.bias.copy_(Wn[1024])
net = net.to(DEV)
yvar = Yt.var(0).sum()
def r2():
    with torch.no_grad():
        e = sum(((net(Ut[a:a+20000]) - Yt[a:a+20000]).pow(2).sum()) for a in range(0, len(Ut), 20000))
        return float(1 - e / len(Ut) / yvar)
out = {"start": r2()}; print(out, flush=True)
params = {"all": list(net.parameters()), "writes": [net.l2.weight, net.l2.bias], "bias": [net.l1.bias],
          "reads": [net.l1.weight, net.l1.bias]}[MODE]
for p_ in net.parameters():
    p_.requires_grad_(any(p_ is q for q in params))
opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.0)
gen = torch.Generator().manual_seed(0)
for ep in range(EP):
    perm = torch.randperm(len(U), generator=gen)
    for i in range(len(U) // 4096):
        j = perm[i*4096:(i+1)*4096]
        loss = (net(U[j].float().to(DEV)) - Y[j].float().to(DEV)).pow(2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if ep in (4, 9, 19, 29) or ep == EP - 1:
        out[f"epoch{ep+1}"] = r2(); print(ep + 1, out[f"epoch{ep+1}"], flush=True)
json.dump(out, open(os.path.join(D, f"ft_{MODE}.json"), "w"), indent=1)
torch.save(dict(state=net.state_dict(), sel=sel, W1=W1, b1=b1, Wn=Wn), os.path.join(D, f"ft_{MODE}.pt"))
