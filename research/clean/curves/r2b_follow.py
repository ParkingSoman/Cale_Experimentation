"""R2b (NEURON-MATH Statement 7.3): write = sum_n a_n W_.n, a = regression of the unit's pooled activity on all 480 pooled
3b activities (FITTED premise check, 2000 train images). Baselines: trained write; in-span lsq write (Prop. 7.2)."""
import numpy as np, torch
from inception import *
from family import Data, IN4A, pool3
from w_fisher import load_build, real_kl, A_cols

m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"])
Wall = [getattr(m, n).weight.detach()[:, :, 0, 0] for n in IN4A]            # (out_j, 480)
A = A_cols(m, fam)
tr = np.sort(np.random.default_rng(5).choice(D.tr, 2000, replace=False))
TAGS = ["L-k1-m379-lr3e-2", "L-k1-r-lr3e-2", "L-k3-m-lr3e-2"]
FMS = {t: load_build(m, fam, t) for t in TAGS}
G = torch.zeros(481, 481, dtype=torch.float64, device="cpu"); R = {t: 0 for t in TAGS}
Gc = {t: 0 for t in TAGS}; Uc = {t: 0 for t in TAGS}; mu_u = 0; mu_c = {t: 0 for t in TAGS}; n = 0
with torch.no_grad():
    for i in range(0, len(tr), 50):
        ins = inputs_3b(m, to_3a(m, prep(D.X[tr[i:i + 50]])))
        x = pool3(F.relu(pre_3b(m, ins[0])))                                  # (B, 480, 14, 14)
        H = torch.cat([x.permute(0, 2, 3, 1).reshape(-1, 480), torch.ones(x.shape[0] * 196, 1, device=DEV)], 1).cpu().double()
        G += H.T @ H; n += H.shape[0]
        hf = H[:, fam]                                                         # family pooled activity
        for t in TAGS:
            c = pool3(F.relu(FMS[t].pre(ins))).permute(0, 2, 3, 1).reshape(-1, FMS[t].k).cpu().double()
            R[t] = R[t] + H.T @ c
            c1 = torch.cat([c, torch.ones(len(c), 1, dtype=torch.float64)], 1)
            Gc[t] = Gc[t] + c1.T @ c1; Uc[t] = Uc[t] + c1.T @ hf             # for the in-span write: regress h_fam on [c, 1]
G_reg = G + 1e-6 * torch.trace(G) / 481 * torch.eye(481, dtype=torch.float64)
builds = {}
for t in TAGS:
    fm = FMS[t]; k = fm.k; zero = [torch.zeros(w.shape[0], device=DEV) for w in Wall]
    a = torch.linalg.solve(G_reg, R[t])[:480].float().to(DEV)               # (480, k)
    B73 = [w @ a for w in Wall]
    beta = torch.linalg.solve(Gc[t] + 1e-9 * torch.eye(k + 1, dtype=torch.float64), Uc[t]).float().to(DEV)   # (k+1, 10)
    Bin = [Aj @ beta[:k].T for Aj in A]; cbin = [Aj @ beta[k] for Aj in A]
    fam_share = (a[fam].pow(2).sum(0) / a.pow(2).sum(0)).cpu().numpy()
    top = torch.argsort(a[:, 0].abs(), descending=True)[:8].cpu().numpy()
    print(f"{t}: share of a^2 on the family {np.round(fam_share, 3)}; top |a| neurons (unit 0) {top.tolist()}", flush=True)
    cos = [float(torch.nn.functional.cosine_similarity(torch.cat(B73, 0)[:, j], torch.cat(fm.weights()[2], 0)[:, j], 0)) for j in range(k)]
    print(f"   cos(B 7.3, trained) per unit {np.round(cos, 3)}", flush=True)
    builds[f"{t} trained"] = (fm, fm.weights()[2], zero)
    builds[f"{t} 7.3"] = (fm, B73, zero)
    builds[f"{t} in-span lsq"] = (fm, Bin, cbin)
K = real_kl(m, fam, D, D.dv, builds)
for t in TAGS:
    tt = K[f"{t} trained"]; print(f"{t}: trained {tt.mean():.4f}")
    for v in ["7.3", "in-span lsq"]:
        d = K[f"{t} {v}"] - tt; print(f"   {v:12s} {K[f'{t} {v}'].mean():.4f}  paired vs trained {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})")
    d = K[f"{t} 7.3"] - K[f"{t} in-span lsq"]; print(f"   7.3 - in-span: {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})")
np.savez("cache/r2b_follow.npz", **{k.replace(' ', '_'): v for k, v in K.items()})
