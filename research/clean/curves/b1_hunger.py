"""B1: Phi (LS read of C1 moved sum + one Fisher write solve) from two disjoint 3500-image halves of the train split.
Dev-5000 KL paired; KL on 1000 of each build's own images (over-fit check). Prediction in CURVES-LOG.md (MIX, B1)."""
import numpy as np, torch, time
from inception import *
from family import Family, Data
from moments import compile_read, read_from
from w_fisher import solve, real_kl
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"])
for p in m.parameters(): p.requires_grad_(False)
perm = np.random.default_rng(31).permutation(D.tr); halves = {"h1": np.sort(perm[:3500]), "h2": np.sort(perm[3500:7000])}
ev, own = {}, {}
for name, ix in halves.items():
    t = time.time(); K, b = read_from(compile_read(m, D.X, fam, ix))
    fm = Family(m, fam, 1, ks=K.shape[-1]).to(DEV); _, _, c0 = fm.weights()
    fm.set_weights(K[None].to(DEV), torch.tensor([b], device=DEV), [x.to(DEV) for x in c0])
    cols, cbs, cov = solve(m, fam, D, fm, ix, 10, 1); ev[name] = (fm, cols, cbs)
    print(f"{name}: built in {time.time() - t:.0f}s, class mass {cov:.3f}", flush=True)
    torch.save(dict(K=K, b=b, cols=[c.cpu() for c in cols], cbs=[c.cpu() for c in cbs]), f"cache/b1_{name}.pt")
Kl = real_kl(m, fam, D, D.dv, ev)
for k, v in Kl.items(): print(f"{k} dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})", flush=True)
d = Kl["h1"] - Kl["h2"]; print(f"paired h1 - h2: {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f}); per-image corr {np.corrcoef(Kl['h1'], Kl['h2'])[0, 1]:.3f}")
for name, ix in halves.items():
    o = real_kl(m, fam, D, ix[:1000], {name: ev[name]})[name]; x = real_kl(m, fam, D, halves["h2" if name == "h1" else "h1"][:1000], {name: ev[name]})[name]
    print(f"{name}: KL on own images {o.mean():.4f}  on the other half's images {x.mean():.4f}", flush=True)
np.savez("cache/b1_dev.npz", **Kl)
