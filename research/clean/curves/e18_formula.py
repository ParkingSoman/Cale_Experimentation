"""E17: Fisher-metric columns (step0 and step1) + plain-mean constant, network-made inputs only; M read and self read."""
import numpy as np, torch
from inception import *
from family import Data, Family
from w_fisher import load_build, solve, real_kl, intact, coarse, A_cols
from u4_w3 import solve_active_k, lsq_write
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); M = load_build(m, fam, "L-k3-m-lr3e-2")
for p in m.parameters(): p.requires_grad_(False)
class Src:
    def __init__(self, X): self.X = X
X = np.load("cache/e15_aug.npy"); SET = Src(np.concatenate([X, X[:, :, ::-1]])); INV = np.load("cache/e1_self_256.npy"); A = A_cols(m, fam)
L = torch.tensor(np.load("cache/e14_what_e9.npz")["top_L"][0], dtype=torch.float32, device=DEV); P = 801
G = torch.zeros(3, P, P, dtype=torch.float64); r = torch.zeros(3, P, dtype=torch.float64)
RFIT = np.concatenate([INV, SET.X[::4]])
with torch.no_grad():
    for i in range(0, len(RFIT), 16):
        ins = inputs_3b(m, to_3a(m, prep(RFIT[i:i + 16]))); Xu = F.unfold(F.pad(ins[2], (2, 2, 2, 2)), 5); Wd = torch.cat([Xu, torch.ones_like(Xu[:, :1])], 1).transpose(1, 2).reshape(-1, P)
        T = F.relu(pre_3b(m, ins[0])[:, fam]).permute(0, 2, 3, 1).reshape(-1, 10) @ L.T
        for j in range(3): o = T[:, j] > 0; Xs = Wd[o]; G[j] += (Xs.T @ Xs).cpu().double(); r[j] += (Xs.T @ T[o, j]).cpu().double()
V = torch.stack([torch.linalg.solve(G[j] + 1e-7 * torch.trace(G[j]) / P * torch.eye(P, dtype=torch.float64), r[j]) for j in range(3)]).float().to(DEV)
S = Family(m, fam, 3).to(DEV); S.set_weights(V[:, :800].reshape(3, 32, 5, 5), V[:, 800], [c.to(DEV) for c in M.weights()[2]])
def plain_beta(fm, cols, idx):
    Sc, Sz, n = [0, 0], [0, 0], [0, 0]
    with torch.no_grad():
        for i in range(0, len(idx), 50):
            ins = inputs_3b(m, to_3a(m, prep(SET.X[idx[i:i + 50]]))); _, z1, z2 = intact(m, fam, ins); c1, c2 = coarse(fm, ins)
            for j, (z, c) in enumerate([(z1, c1), (z2, c2)]): Sz[j] = Sz[j] + z.sum((0, 2, 3)); Sc[j] = Sc[j] + c.sum((0, 2, 3)); n[j] += z.shape[0] * z.shape[2] * z.shape[3]
    return [A[j] @ (Sz[min(j // 3, 1)] / n[min(j // 3, 1)]) - cols[j] @ (Sc[min(j // 3, 1)] / n[min(j // 3, 1)]) for j in range(4)]
idx = np.arange(len(SET.X)); sub = idx[::4]; B = {}
for tag, fm in [("self read (fit on the set)", S)]:
    c0, b0, _ = solve(m, fam, SET, fm, idx, 10, 3); B[f"{tag}: step0 + plain beta"] = (fm, c0, plain_beta(fm, c0, sub))
    c1, b1, _ = solve_active_k(m, fam, SET, fm, idx, (c0, b0), 3); B[f"{tag}: step1 + plain beta"] = (fm, c1, plain_beta(fm, c1, sub))
    B[f"{tag}: lsq on the set"] = (fm, *lsq_write(m, fam, SET, fm, sub, 3)); print("done", tag, flush=True)
B["self read (fit on the set): lsq on class inversions"] = (S, *lsq_write(m, fam, Src(INV), S, np.arange(256), 3))
K = real_kl(m, fam, D, D.dv, B); ref = np.load("cache/u9_kl_35000.npz")
for k, v in K.items(): print(f"  {k:42s} dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})")
e4 = np.load("cache/e17_formula.npz")["self_read_lsq_on_class_inversions_(E4)"]
for k in K:
    d = K[k] - e4; print(f"  paired {k} - E4 (.0771): {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})")
for k in K:
    d = K[k] - ref["M_read_+_trained_write"]; print(f"  {k} vs bar B {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f}); vs THE FORMULA (FITTED) {(K[k] - ref['arcs_read_+_step1']).mean():+.4f}")
np.savez("cache/e18_formula.npz", **{k.replace(' ', '_').replace(':', ''): v for k, v in K.items()})
