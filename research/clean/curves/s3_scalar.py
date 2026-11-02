"""S3: identity lsq vs s-weighted lsq (s = per-position output sensitivity) vs Fisher step0, real unsure 2000, M read."""
import numpy as np, torch
from inception import *
from family import Data
import w_fisher as W
from w_fisher import load_build, real_kl, intact, coarse, A_cols, fisher_grads
from scipy.special import softmax
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); M = load_build(m, fam, "L-k3-m-lr3e-2"); A = A_cols(m, fam)
for p in m.parameters(): p.requires_grad_(False)
P = softmax(np.asarray(D.T[D.tr], np.float64), 1); uns = D.tr[1 - (P ** 2).sum(1) > .1]; idx = np.sort(np.random.default_rng(81).choice(uns, 2000, replace=False))
def wlsq(weighted):
    acc = [dict(S=0, SS=0, N=0) for _ in range(2)]
    for i in range(0, len(idx), 25):
        with torch.no_grad(): ins = inputs_3b(m, to_3a(m, prep(D.X[idx[i:i + 25]]))); outs, z1, z2 = intact(m, fam, ins); c1, c2 = coarse(M, ins)
        if weighted:
            G, _ = fisher_grads(m, outs, 10); s = sum(sum((g[j] ** 2).sum(1) for j in range(4)) for g in G).detach()   # (B, 14, 14)
        else:
            s = torch.ones_like(c1[:, 0])
        with torch.no_grad():
            for a, (z, c) in zip(acc, [(z1, c1), (z2, c2)]):
                x = torch.cat([z, c, torch.ones_like(c[:, :1])], 1).permute(0, 2, 3, 1).reshape(-1, 14).cpu().double(); w = s.reshape(-1).cpu().double()
                a["S"] = a["S"] + (w[:, None] * x).sum(0); a["SS"] = a["SS"] + (w[:, None] * x).T @ x; a["N"] += float(w.sum())
    cols, cbs = [], []
    for j in range(4):
        a = acc[1] if j == 3 else acc[0]; SS = (a["SS"] / a["N"]).numpy(); Aj = A[j].cpu().double().numpy()
        Cx = SS[10:, 10:]; Czx = SS[:10, 10:]; sol = np.linalg.solve(Cx + 1e-9 * np.trace(Cx) * np.eye(4), (Aj @ Czx).T).T   # regress A z on [c, 1]
        cols.append(torch.tensor(sol[:, :3], dtype=torch.float32, device=DEV)); cbs.append(torch.tensor(sol[:, 3], dtype=torch.float32, device=DEV))
    return cols, cbs
B = {"identity lsq": (M, *wlsq(False)), "s-weighted lsq": (M, *wlsq(True))}
c0, b0, _ = W.solve(m, fam, D, M, idx, 10, 3); B["Fisher step0"] = (M, c0, b0)
K = real_kl(m, fam, D, D.dv, B)
for k, v in K.items(): print(f"  {k:16s} dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})")
gap = K["identity lsq"].mean() - K["Fisher step0"].mean(); rec = K["identity lsq"].mean() - K["s-weighted lsq"].mean()
print(f"  share of the lsq -> Fisher gap recovered by the scalar weighting: {rec / gap:.2f}")
