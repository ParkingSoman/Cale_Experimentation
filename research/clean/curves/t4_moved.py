"""T2/T3 instrument fix: moved children (C1 shifts, per member, before pooling) and lam >= 0 (exact by supports)."""
import itertools, numpy as np, torch
from inception import *
from family import pool3
from w_fisher import load_build
from moments import c1_shifts
m = load_model(); X, y = load_images(); tr, dv, se = split(y); fam = list(np.load("cache/s1_fine.npz")["fam"])
s, _ = c1_shifts(); Phi = np.load("cache/t2.npz")["Phi"]
def move(z):
    return torch.stack([torch.roll(F.pad(z[:, i], (8, 8, 8, 8)), shifts=(-int(s[i][0]), -int(s[i][1])), dims=(1, 2))[:, 8:36, 8:36] for i in range(10)], 1)
flat = lambda t: t[:, :, 2:12, 2:12].permute(1, 0, 2, 3).reshape(t.shape[1], -1).cpu().double().numpy()
ix = np.sort(np.random.default_rng(41).permutation(tr)[:1500]); fms = {t: load_build(m, fam, t) for t in ["L-k1-r-lr3e-2", "fam-curves-k1-lr0.03-s0-random"]}
Z, Zm, Cs = [], [], {t: [] for t in fms}
with torch.no_grad():
    for i in range(0, len(ix), 50):
        ins = inputs_3b(m, to_3a(m, prep(X[ix[i:i + 50]]))); h = F.relu(pre_3b(m, ins[0]))[:, fam]
        Z.append(flat(pool3(h))); Zm.append(flat(pool3(move(h))))
        if i < 500:
            for t, fm in fms.items(): Cs[t].append(flat(pool3(F.relu(fm.pre(ins))))[0])
Z, Zm = np.concatenate(Z, 1), np.concatenate(Zm, 1); n = Z.shape[1]
S_all = np.cov(np.concatenate([Z, Zm]), bias=True); Szz, Szm, Smm = S_all[:10, :10], S_all[:10, 10:], S_all[10:, 10:]
mu = Z.mean(1); tot = np.trace(Phi @ (Szz + np.outer(mu, mu))); Q = Szm.T @ Phi @ Szm
best = (-1, None)
for k in range(1, 11):
    for sup in itertools.combinations(range(10), k):
        sup = list(sup); ev, V = np.linalg.eig(np.linalg.solve(Smm[np.ix_(sup, sup)], Q[np.ix_(sup, sup)]))
        j = np.argmax(ev.real); v = V[:, j].real; v = v * np.sign(v.sum())
        if (v >= -1e-12).all() and ev[j].real > best[0]:
            lam = np.zeros(10); lam[sup] = v; best = (ev[j].real, lam)
val, lam = best; lam /= lam.sum()
ev, V = np.linalg.eig(np.linalg.solve(Smm, Q)); free = V[:, np.argmax(ev.real)].real; free *= np.sign(free.sum()); free /= np.abs(free).sum()
np.set_printoptions(precision=3, suppress=True, linewidth=150)
print(f"shares: beta {mu @ Phi @ mu / tot:.3f} | constrained neuron {val / tot:.3f} | unconstrained neuron {ev.real.max() / tot:.3f}")
print("lam constrained (moved)", lam); print("lam free (moved)       ", free)
nm = Z.shape[1] // 3   # the first 500 images
for t in fms:
    c = np.concatenate(Cs[t]); out = {}
    for name, M in [("unmoved", Z[:, :len(c)]), ("moved", Zm[:, :len(c)])]:
        D = np.concatenate([M.T, np.ones((len(c), 1))], 1); w = np.linalg.lstsq(D, c, rcond=None)[0]
        out[name] = (1 - ((c - D @ w) ** 2).mean() / c.var(), w[:10] / np.abs(w[:10]).sum())
    lt = out["moved"][1]
    print(f"{t}: S1 R^2 unmoved {out['unmoved'][0]:.2f} moved {out['moved'][0]:.2f} | lam_trained(moved) {lt} | S2 corr constrained {np.corrcoef(lt, lam)[0, 1]:.2f}, free {np.corrcoef(lt, free)[0, 1]:.2f}")
print("twin agreement on moved lam:", np.corrcoef(*[None for _ in []]) if False else "")
np.savez("cache/t4.npz", lam=lam, free=free, Szz=Szz, Szm=Szm, Smm=Smm, mu=mu, mum=Zm.mean(1))
