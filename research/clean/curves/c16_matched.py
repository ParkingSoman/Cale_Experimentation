"""22.2 positive check: per-activity-image metric from other train images of the SAME label vs random ones."""
import numpy as np, torch
src = open("c11_exactavg.py").read(); src = src.split("Lf = {k: 0.0")[0].replace("G = torch.cat([reader_grads(Xm[i:i + 20]).flatten(1).half() for i in range(0, 500, 20)])", "G = None").replace("print(f\"G {tuple(G.shape)}", "print(f\"G skipped")
exec(src)
rng = np.random.default_rng(12); y = D.y; tr = D.tr
same = np.array([rng.choice(np.setdiff1d(tr[y[tr] == y[x]], [x]), 2, replace=False) for x in Xa])
rand = np.array([rng.choice(np.setdiff1d(tr, [x]), 2, replace=False) for x in Xa])
res = {"matched": {k: 0.0 for k in builds}, "random": {k: 0.0 for k in builds}}
for i in range(0, 500, 10):
    b = Xa[i:i + 10]; x, p = family_inputs(b)
    with torch.no_grad():
        a0 = pre(x, p); E = {k: (a0 - pre(x, p, fm)).flatten(1) for k, fm in builds.items()}
    for tag, M in [("matched", same), ("random", rand)]:
        mi = M[i:i + 10].ravel(); o = np.argsort(mi); Gm = torch.empty(len(mi), 10, 72128, device=DEV)
        Gm[torch.from_numpy(o).to(DEV)] = reader_grads(mi[o]).reshape(len(mi), 10, -1)
        Gm = Gm.reshape(10, 2, 10, -1)                                                     # (activity image, partner, class, dim)
        for k in builds: res[tag][k] += float((torch.einsum("apcd,ad->apc", Gm, E[k]) ** 2).sum()) / 2 / 2
for tag in res:
    v = {k: s / 500 for k, s in res[tag].items()}
    print(f"{tag:8s} " + "  ".join(f"{k} {x:.4f}" for k, x in v.items()) + f"   S+rem/S {v['S+rem'] / v['S']:.3f}  T/S {v['T'] / v['S']:.3f}", flush=True)
