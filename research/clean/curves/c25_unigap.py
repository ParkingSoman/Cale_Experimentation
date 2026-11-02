"""Does the per-image second-order objective with a UNIFORM output metric credit training's move (Gap B)?"""
import numpy as np, torch
src = open("c17_fisher.py").read(); exec(src.split("G = torch.zeros(NRD, NRD")[0])
def reader_grads_uni(b, rep=10):
    x, p = family_inputs(b); post = [t.requires_grad_(True) for t in readers_post(x, p)]
    a_, b_, c_, d_ = post
    lg = rest_from_4a(torch.cat([a_, F.relu(C3(F.pad(b_, (1, 1, 1, 1)))), F.relu(C5(F.pad(c_, (2, 2, 2, 2)))), d_], 1))
    out = []
    for r in range(rep):
        xi = torch.randn_like(lg); xi = xi - xi.mean(1, keepdim=True)
        out.append(torch.cat(torch.autograd.grad((lg * xi).sum(), post, retain_graph=r < rep - 1), 1))
    return torch.stack(out, 1).detach()                                               # (B, rep, 368, 14, 14)
res = {"class": {k: 0.0 for k in Vs}, "uniform": {k: 0.0 for k in Vs}}
for i in range(0, 500, 10):
    b = TR[i:i + 10]; y = ybatch(b)
    for tag, g in [("class", reader_grads(b).reshape(len(b), 10, NRD, 196)), ("uniform", reader_grads_uni(b).reshape(len(b), 10, NRD, 196))]:
        Gm = torch.einsum("bcjp,bkp->bcjk", g, y) * mask
        for k, V in Vs.items(): res[tag][k] += float(((Gm * V).sum((2, 3)) ** 2).sum())
for tag, d in res.items():
    print(f"{tag:8s} S+rem/S {d['S+rem'] / d['S']:.3f}   T/S {d['T'] / d['S']:.3f}   4d/S {d['4d'] / d['S']:.3f}")
