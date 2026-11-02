"""BEAT I1-I2: what training made at k = 3. Regress each unit's pooled output on the pooled children (T4 images).
Predictions in CURVES-LOG.md (BEAT, I1-I2)."""
import numpy as np, torch
from inception import *
from family import pool3
from w_fisher import load_build
from moments import c1_shifts
m = load_model(); X, y = load_images(); tr, dv, se = split(y); fam = list(np.load("cache/s1_fine.npz")["fam"])
PREF = np.deg2rad([10, 50, 70, 120, 170, 180, 250, 290, 320, 350])    # s1 order
s, _ = c1_shifts()
def move(z):
    return torch.stack([torch.roll(F.pad(z[:, i], (8, 8, 8, 8)), shifts=(-int(s[i][0]), -int(s[i][1])), dims=(1, 2))[:, 8:36, 8:36] for i in range(10)], 1)
flat = lambda t: t[:, :, 2:12, 2:12].permute(1, 0, 2, 3).reshape(t.shape[1], -1).cpu().double().numpy()
ix = np.sort(np.random.default_rng(41).permutation(tr)[:1500])[:500]
tags = ["L-k1-r-lr3e-2", "L-k3-r-lr3e-2", "L-k3-m-lr3e-2"]; fms = {t: load_build(m, fam, t) for t in tags}
Z, Zm, Cs = [], [], {t: [] for t in tags}
with torch.no_grad():
    for i in range(0, len(ix), 50):
        ins = inputs_3b(m, to_3a(m, prep(X[ix[i:i + 50]]))); h = F.relu(pre_3b(m, ins[0]))[:, fam]
        Z.append(flat(pool3(h))); Zm.append(flat(pool3(move(h))))
        for t, fm in fms.items(): Cs[t].append(flat(pool3(F.relu(fm.pre(ins)))))
Z, Zm = np.concatenate(Z, 1), np.concatenate(Zm, 1); Cs = {t: np.concatenate(v, 1) for t, v in Cs.items()}
np.set_printoptions(precision=3, suppress=True, linewidth=160)
def reg(M, c):
    D = np.concatenate([M.T, np.ones((M.shape[1], 1))], 1); w = np.linalg.lstsq(D, c, rcond=None)[0]
    return 1 - ((c - D @ w) ** 2).mean() / c.var(), w[:10]
corr = lambda a, b: np.corrcoef(a, b)[0, 1]
lam1 = reg(Zm, Cs["L-k1-r-lr3e-2"][0])[1]; lam1u = reg(Z, Cs["L-k1-r-lr3e-2"][0])[1]
print("instrument (k=1 twin): R^2 unmoved %.2f moved %.2f" % (reg(Z, Cs["L-k1-r-lr3e-2"][0])[0], reg(Zm, Cs["L-k1-r-lr3e-2"][0])[0]))
print("k=1 lam unmoved", lam1u / np.abs(lam1u).sum())
out = {}
for t in tags[1:]:
    C = Cs[t]; print(f"\n== {t}: on-rate per unit", (C > 0).mean(1), "| pairwise corr of pooled outputs:",
                     [round(corr(C[a], C[b]), 2) for a, b in [(0, 1), (0, 2), (1, 2)]])
    for j in range(3):
        r2u, lu = reg(Z, C[j]); r2m, lm = reg(Zm, C[j]); lu_n = lu / np.abs(lu).sum()
        print(f" unit {j}: R^2 unmoved {r2u:.2f} moved {r2m:.2f} | lam unmoved {lu_n} | corr with k=1 lam (unmoved) {corr(lu, lam1u):.2f}")
        out[(t, j)] = lu
# I2: bumps around start members (M)
start = [0, 3, 6]   # 330, 388, 385 in s1 order
print("\nI2 (unmoved lam of M units vs shapes)")
for j, st in enumerate(start):
    l = out[("L-k3-m-lr3e-2", j)]; phi = PREF[st]; ang = np.angle(np.exp(1j * (PREF - phi)))
    bump = 1 + np.cos(ang); one = (np.arange(10) == st).astype(float)
    nb = np.zeros(10); nb[[st, (st - 1) % 10, (st + 1) % 10]] = 1
    # best-fitting phase of 1 + cos (free phase), and best width via cos^p? no: report fixed-shape scores only
    print(f" unit {j} start {np.rad2deg(phi):.0f} deg: corr bump {corr(l, bump):.2f} | one-hot {corr(l, one):.2f} | member+nbrs {corr(l, nb):.2f}")
    print("   lam by member angle:", dict(zip(np.rad2deg(PREF).astype(int), np.round(l / np.abs(l).sum(), 3))))
np.savez("cache/u1_k3.npz", **{f"{t}_{j}": v for (t, j), v in out.items()}, Z=Z[:, :20000], lam1u=lam1u)
