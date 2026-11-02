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
with torch.no_grad():
    for i in range(0, len(INV), 16):
        ins = inputs_3b(m, to_3a(m, prep(INV[i:i + 16]))); Xu = F.unfold(F.pad(ins[2], (2, 2, 2, 2)), 5); Wd = torch.cat([Xu, torch.ones_like(Xu[:, :1])], 1).transpose(1, 2).reshape(-1, P)
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
idx = np.arange(len(SET.X)); sub = idx[::4]; INVS = Src(INV)
def plain_beta_on(src, fm, cols, ii):
    Sc, Sz, n = [0, 0], [0, 0], [0, 0]
    with torch.no_grad():
        for i in range(0, len(ii), 50):
            ins = inputs_3b(m, to_3a(m, prep(src.X[ii[i:i + 50]]))); _, z1, z2 = intact(m, fam, ins); c1, c2 = coarse(fm, ins)
            for j, (z, c) in enumerate([(z1, c1), (z2, c2)]): Sz[j] = Sz[j] + z.sum((0, 2, 3)); Sc[j] = Sc[j] + c.sum((0, 2, 3)); n[j] += z.shape[0] * z.shape[2] * z.shape[3]
    return [A[j] @ (Sz[min(j // 3, 1)] / n[min(j // 3, 1)]) - cols[j] @ (Sc[min(j // 3, 1)] / n[min(j // 3, 1)]) for j in range(4)]
B = {}
for tag, fm in [("self read", S), ("M read", M)]:
    c0, b0, _ = solve(m, fam, SET, fm, idx, 10, 3); c1, b1, _ = solve_active_k(m, fam, SET, fm, idx, (c0, b0), 3)
    ls_set, _ = lsq_write(m, fam, SET, fm, sub, 3); ls_inv, _ = lsq_write(m, fam, INVS, fm, np.arange(256), 3)
    comb = [a + (f - b) for a, f, b in zip(ls_inv, c1, ls_set)]
    B[f"{tag}: lsq(inv) + [Fisher(set) - lsq(set)]"] = (fm, comb, plain_beta_on(INVS, fm, comb, np.arange(256)))
    B[f"{tag}: lsq(inv)"] = (fm, ls_inv, plain_beta_on(INVS, fm, ls_inv, np.arange(256)))
    B[f"{tag}: Fisher(set) + plain beta(set)"] = (fm, c1, plain_beta_on(SET, fm, c1, sub)); print("done", tag, flush=True)
K = real_kl(m, fam, D, D.dv, B); ref = np.load("cache/u9_kl_35000.npz"); e4 = np.load("cache/e17_formula.npz")["self_read_lsq_on_class_inversions_(E4)"]
for k, v in K.items():
    d = K[k] - e4; b = K[k] - ref["M_read_+_trained_write"]
    print(f"  {k:48s} dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f}) | vs E4 {d.mean():+.4f} ({d.std() / np.sqrt(len(d)):.4f}) | vs bar B {b.mean():+.4f}")
np.savez("cache/e19_split.npz", **{k.replace(' ', '_').replace(':', ''): v for k, v in K.items()})
