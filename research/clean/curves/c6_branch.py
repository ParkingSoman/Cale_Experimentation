"""B4 (Part VI): fit one Delta-Lambda per branch (edge) to training's local move T - S; energy explained and dev-5000 KL
of S + per-branch part. Baseline: the same fit with a random 4-way split of the 368 readers (same sizes)."""
import numpy as np, torch, copy
from circ import *

S = b3_train.FamilyRA(m, fam, 1); S.add_rb(m); S.add_ra(m); S.load_state_dict(torch.load("cache/x5f_factor_n2000.pt", map_location="cpu")); S = S.to(DEV)
T = b3_train.load_b3(m, "B3-k0-factorlocal"); cs = [float(c) for c in S.cscale]
dW = torch.cat([(T.dW[j] - S.dW[j]).detach() * cs[j] for j in range(4)]).cpu().double(); W = torch.cat([w.cpu() for w in WF]).double()
sizes = [w.shape[0] for w in WF]; br = np.repeat(np.arange(4), sizes)
def fit(labels):
    P = torch.zeros_like(dW)
    for b in range(4):
        s = torch.from_numpy(labels == b); P[s] = W[s] @ torch.linalg.lstsq(W[s], dW[s]).solution
    return P
tot = float((dW ** 2).sum()); Pb = fit(br)
print(f"per-branch: explains {float((Pb ** 2).sum()) / tot:.3f} of |Delta-W|^2 (residual {float(((dW - Pb) ** 2).sum()) / tot:.3f})")
rnd = [float(((fit(np.random.default_rng(s).permutation(br)) ** 2).sum())) / tot for s in range(5)]
print(f"random 4-way split (5 seeds): explains {np.mean(rnd):.3f} +- {np.std(rnd):.3f}")
for b in range(4):
    s = br == b; print(f"  branch {b} ({sizes[b]} readers): share of |Delta-W|^2 {float((dW[s] ** 2).sum()) / tot:.3f}, explained within branch {float((Pb[s] ** 2).sum() / (dW[s] ** 2).sum()):.3f}")
fm = copy.deepcopy(S); off = np.cumsum([0] + sizes)
with torch.no_grad():
    for j in range(4): fm.dW[j] += Pb[off[j]:off[j + 1]].float().to(DEV) / cs[j]
r = per_image(fm); s0 = np.load("cache/c5_gapb.npz")
print(f"S+branch {r.mean():.4f} (se {r.std() / np.sqrt(len(r)):.4f})")
for k in ["S", "S+rem", "T"]:
    d = r - s0[k]; print(f"  S+branch minus {k} {d.mean():+.4f} (paired se {d.std() / np.sqrt(len(d)):.4f})")
np.save("cache/c6_branch.npy", r)
