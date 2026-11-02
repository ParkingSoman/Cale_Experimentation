"""B1 part (3) redone: the first version took ix[:1000] of a class-sorted file (all animals). Random 1000 instead.
Also splits the saved dev-5000 KL by mix (animals = classes 0-397)."""
import numpy as np, torch
from inception import *
from family import Family, Data
from w_fisher import real_kl
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"])
Kl = dict(np.load("cache/b1_dev.npz")); ani = D.y[D.dv] < 398
for k, v in Kl.items(): print(f"{k} dev KL animals {v[ani].mean():.4f} (se {v[ani].std() / np.sqrt(ani.sum()):.4f})  rest {v[~ani].mean():.4f} (se {v[~ani].std() / np.sqrt((~ani).sum()):.4f})")
perm = np.random.default_rng(31).permutation(D.tr); halves = {"h1": np.sort(perm[:3500]), "h2": np.sort(perm[3500:7000])}
rest = np.setdiff1d(D.tr, np.concatenate(list(halves.values()))); ev = {}
for name in halves:
    ck = torch.load(f"cache/b1_{name}.pt"); fm = Family(m, fam, 1, ks=ck["K"].shape[-1]).to(DEV); _, _, c0 = fm.weights()
    fm.set_weights(ck["K"][None].to(DEV), torch.tensor([ck["b"]], device=DEV), [x.to(DEV) for x in c0])
    ev[name] = (fm, [c.to(DEV) for c in ck["cols"]], [c.to(DEV) for c in ck["cbs"]])
r = np.random.default_rng(7)
sets = {"own": None, "unused-train": np.sort(r.choice(rest, 1000, replace=False))}
u = real_kl(m, fam, D, sets["unused-train"], ev)
for name, ix in halves.items():
    o = real_kl(m, fam, D, np.sort(r.choice(ix, 1000, replace=False)), {name: ev[name]})[name]
    print(f"{name}: own 1000 {o.mean():.4f} (se {o.std() / np.sqrt(1000):.4f})  unused train 1000 {u[name].mean():.4f} (se {u[name].std() / np.sqrt(1000):.4f})  dev {Kl[name].mean():.4f}", flush=True)
