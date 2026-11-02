"""T1: Theorems 10-11 on the curve family, F = identity. Weights (4a columns) + C(nu) (10 x 10) only."""
import numpy as np, torch
from inception import *
from family import IN4A, pool3
from derive import PREF
m = load_model(); X, y = load_images(); tr, dv, se = split(y); fam = list(np.load("cache/s1_fine.npz")["fam"])
A = np.concatenate([getattr(m, n).weight.detach()[:, fam, 0, 0].cpu().numpy() for n in IN4A]).astype(np.float64)  # (368, 10)
def C_of(ix):
    C = np.zeros((10, 10)); n = 0
    with torch.no_grad():
        for i in range(0, len(ix), 50):
            z = pool3(F.relu(pre_3b(m, to_3a(m, prep(X[np.sort(ix[i:i + 50])])))[:, fam]))[:, :, 2:12, 2:12]
            z = z.permute(1, 0, 2, 3).reshape(10, -1).cpu().double().numpy(); C += z @ z.T; n += z.shape[1]
    return C / n
r = np.random.default_rng(41); g = r.permutation(tr)
mixes = {"gen": g[:1500], "gen2": g[1500:3000], "ani": r.permutation(tr[y[tr] < 398])[:1500], "rest": r.permutation(tr[y[tr] >= 398])[:1500]}
Cs = {k: C_of(v) for k, v in mixes.items()}
AA = A.T @ A; np.set_printoptions(precision=3, suppress=True)
print("P1: min entry of A^T A (normalised) %.3f ; min entry of C (normalised) %.3f" % ((AA / np.sqrt(np.outer(np.diag(AA), np.diag(AA)))).min(), min((C / np.sqrt(np.outer(np.diag(C), np.diag(C)))).min() for C in Cs.values())))
print("member pref angles (deg):", np.rad2deg(PREF).round(0), " fam:", fam)
lam = {}
for k, C in Cs.items():
    ev, V = np.linalg.eig(AA @ C); i = np.argmax(ev.real); l = np.abs(V[:, i].real); lam[k] = l / l.sum()
    Y = A @ np.linalg.cholesky(C); U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    Bc = np.stack([np.cos(PREF), np.sin(PREF)], 1); Qb, _ = np.linalg.qr(np.concatenate([np.ones((10, 1)), Bc], 1))
    # right singular vectors live in the C^(1/2)-whitened space; map back to member weights: Lam = C^(-1/2) V (use L^-T)
    L = np.linalg.cholesky(C); Lam = np.linalg.solve(L.T, Vt.T)
    h1 = [np.linalg.norm(Qb[:, 1:].T @ (Lam[:, j] / np.linalg.norm(Lam[:, j]))) ** 2 for j in (1, 2)]
    h0 = np.linalg.norm(Qb[:, :1].T @ (Lam[:, 0] / np.linalg.norm(Lam[:, 0]))) ** 2
    print(f"{k:5s} lam {lam[k]} | min/max {lam[k].min() / lam[k].max():.2f} | sigma^2 share {(S ** 2 / (S ** 2).sum())[:4]} | Lam1 in h0 {h0:.2f} | Lam2, Lam3 in h1 {h1[0]:.2f} {h1[1]:.2f}")
cs = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
print("P4: 1-cos split-half (gen,gen2) %.5f ; 1-cos (ani, rest) %.5f ; weight on member 324: ani %.3f rest %.3f" % (1 - cs(lam["gen"], lam["gen2"]), 1 - cs(lam["ani"], lam["rest"]), lam["ani"][fam.index(324)], lam["rest"][fam.index(324)]))
np.savez("cache/t1_C.npz", A=A, **{"C_" + k: v for k, v in Cs.items()})
