"""BEAT I3: is each trained unit's PRE-activation a linear mix of the members' pre-activations (+ constant)?
If yes, the read lies in the member span and is realisable exactly: kernel sum_i w_i K_i, bias sum_i w_i b_i + b0."""
import numpy as np, torch
from inception import *
from w_fisher import load_build
from moments import c1_shifts
m = load_model(); X, y = load_images(); tr, dv, se = split(y); fam = list(np.load("cache/s1_fine.npz")["fam"])
s, _ = c1_shifts()
def move(z):
    return torch.stack([torch.roll(F.pad(z[:, i], (8, 8, 8, 8)), shifts=(-int(s[i][0]), -int(s[i][1])), dims=(1, 2))[:, 8:36, 8:36] for i in range(10)], 1)
flat = lambda t: t[:, :, 4:24, 4:24].permute(1, 0, 2, 3).reshape(t.shape[1], -1).cpu().double().numpy()
ix = np.sort(np.random.default_rng(41).permutation(tr)[:1500])[:200]
tags = ["L-k1-r-lr3e-2", "fam-curves-k1-lr0.03-s0-random", "L-k3-m-lr3e-2"]; fms = {t: load_build(m, fam, t) for t in tags}
Z, Zm, P = [], [], {t: [] for t in tags}
with torch.no_grad():
    for i in range(0, len(ix), 50):
        ins = inputs_3b(m, to_3a(m, prep(X[ix[i:i + 50]]))); z = pre_3b(m, ins[0])[:, fam]
        Z.append(flat(z)); Zm.append(flat(move(z)))
        for t, fm in fms.items(): P[t].append(flat(fm.pre(ins)))
Z, Zm = np.concatenate(Z, 1), np.concatenate(Zm, 1); P = {t: np.concatenate(v, 1) for t, v in P.items()}
np.set_printoptions(precision=3, suppress=True, linewidth=160)
def reg(M, c, gate=None):
    D = np.concatenate([M.T, np.ones((M.shape[1], 1))], 1)
    if gate is not None: D, c = D[gate], c[gate]
    w = np.linalg.lstsq(D, c, rcond=None)[0]; return 1 - ((c - D @ w) ** 2).mean() / c.var(), w
for t in tags:
    for j in range(P[t].shape[0]):
        c = P[t][j]; on = c > 0
        r_u, w_u = reg(Z, c); r_m, w_m = reg(Zm, c); r_uo, _ = reg(Z, c, on); r_mo, _ = reg(Zm, c, on)
        print(f"{t} unit {j}: pre R^2 unmoved {r_u:.3f} (on-set {r_uo:.3f}) | moved {r_m:.3f} (on-set {r_mo:.3f}) | on {on.mean():.2f}")
        print("   w unmoved", w_u[:10] / np.abs(w_u[:10]).sum(), " b0/|w| %.0f" % (w_u[10] / np.abs(w_u[:10]).sum()))
