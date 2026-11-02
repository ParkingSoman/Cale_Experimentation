"""E2: compare the network's self-made inputs with the generalist images (checks E1-E4 in CURVES-LOG.md, section E).
usage: e2_compare.py N"""
import sys, numpy as np, torch
from inception import *
from family import Data, Family, pool3
from w_fisher import real_kl
from u4_w3 import lsq_write
N = int(sys.argv[1]); m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"])
SELF = np.load(f"cache/e1_self_{N}.npy", mmap_mode="r")
class Self: X = SELF
u = np.load("cache/u5_what.npz"); Lnp = u["top_L"][1]; L = torch.tensor(Lnp, dtype=torch.float32, device=DEV)   # arcs #2
h = torch.load("cache/u6_how.pt", weights_only=False); ci = h["names"].index("arcs #2"); Vreal = h["V"][3 * ci:3 * ci + 3].to(DEV)
P = 801
def windows(ins):
    Xu = F.unfold(F.pad(ins[2], (2, 2, 2, 2)), 5); return torch.cat([Xu, torch.ones_like(Xu[:, :1])], 1).transpose(1, 2).reshape(-1, P)
def stats(X, idx):
    G = torch.zeros(3, P, P, dtype=torch.float64); r = torch.zeros(3, P, dtype=torch.float64); Z = []; on = 0; n = 0
    with torch.no_grad():
        for i in range(0, len(idx), 16):
            ins = inputs_3b(m, to_3a(m, prep(X[idx[i:i + 16]]))); W = windows(ins)
            zr = F.relu(pre_3b(m, ins[0])[:, fam]); z = zr.permute(0, 2, 3, 1).reshape(-1, 10); T = z @ L.T
            for j in range(3):
                o = T[:, j] > 0; Xs = W[o]; G[j] += (Xs.T @ Xs).cpu().double(); r[j] += (Xs.T @ T[o, j]).cpu().double()
            Z.append(pool3(zr)[:, :, 2:12, 2:12].permute(1, 0, 2, 3).reshape(10, -1).cpu().double().numpy())
            on = on + (z > 0).float().sum(0).cpu().numpy(); n += len(z)
    V = torch.stack([torch.linalg.solve(G[j] + 1e-7 * torch.trace(G[j]) / P * torch.eye(P, dtype=torch.float64), r[j]) for j in range(3)]).float().to(DEV)
    return V, np.cov(np.concatenate(Z, 1), bias=True), on / n
Vs, Ss, ons = stats(SELF, np.arange(N))
_, Sr, onr = stats(D.X, np.sort(np.random.default_rng(21).choice(D.tr, 256, replace=False)))
Rs = Ss / np.sqrt(np.outer(np.diag(Ss), np.diag(Ss))); Rr = Sr / np.sqrt(np.outer(np.diag(Sr), np.diag(Sr))); iu = np.triu_indices(10, 1)
np.set_printoptions(precision=3, suppress=True, linewidth=150)
print(f"E1 children corr pattern, Pearson over 45 pairs (self vs real 256): {np.corrcoef(Rs[iu], Rr[iu])[0, 1]:+.3f}")
print("E1 member on-rate self:", ons, "\n            real:", onr, "\n            ratio:", ons / onr)
print("E1 pooled energy share self:", np.diag(Ss) / np.trace(Ss), "\n                        real:", np.diag(Sr) / np.trace(Sr))
print("E2 read kernel cos (self vs real arcs #2):", F.cosine_similarity(Vs[:, :800], Vreal[:, :800], 1).cpu().numpy(),
      "bias self", Vs[:, 800].cpu().numpy(), "real", Vreal[:, 800].cpu().numpy())
def build(V):
    fm = Family(m, fam, 3).to(DEV); _, _, c0 = fm.weights(); fm.set_weights(V[:, :800].reshape(3, 32, 5, 5), V[:, 800], [c.to(DEV) for c in c0]); return fm
fr, fs = build(Vreal), build(Vs)
wimg = np.sort(np.random.default_rng(21).choice(D.tr, 1000, replace=False))
B = {"real read + real lsq": (fr, *lsq_write(m, fam, D, fr, wimg, 3)), "self read + real lsq": (fs, *lsq_write(m, fam, D, fs, wimg, 3)),
     "self read + self lsq (DERIVED)": (fs, *lsq_write(m, fam, Self, fs, np.arange(N), 3))}
K = real_kl(m, fam, D, D.dv, B)
for n, v in K.items():
    d = v - K["real read + real lsq"]; print(f"  {n:32s} dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f}) | paired vs real/real {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})", flush=True)
np.savez(f"cache/e2_compare_{N}.npz", **{n.split()[0] + str(i): v for i, (n, v) in enumerate(K.items())})
