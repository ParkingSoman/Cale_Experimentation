"""E20: Fisher step0 columns on class inversions with the metric at temperature T; plain-mean constant."""
import numpy as np, torch
from inception import *
from family import Data, Family, pool3
import w_fisher as W
from w_fisher import load_build, real_kl, intact, coarse, A_cols
from u4_w3 import lsq_write
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); M = load_build(m, fam, "L-k3-m-lr3e-2")
for p in m.parameters(): p.requires_grad_(False)
class Src:
    def __init__(self, X): self.X = X
INV = np.load("cache/e1_self_256.npy"); rng = np.random.default_rng(121); aug = []
for r in range(4):
    for x in INV:
        dy, dx = rng.integers(-16, 17, 2); p = np.pad(x, ((16, 16), (16, 16), (0, 0)), mode="reflect"); aug.append(p[16 + dy:240 + dy, 16 + dx:240 + dx])
aug = np.stack(aug); SET = Src(np.concatenate([aug, aug[:, :, ::-1]])); INVS = Src(INV); A = A_cols(m, fam)
with torch.no_grad():
    lg = torch.cat([W.upper(m, intact(m, fam, inputs_3b(m, to_3a(m, prep(INV[i:i + 32]))))[0]) for i in range(0, 256, 32)])
    unc = lambda T: float((1 - (F.softmax(lg / T, 1) ** 2).sum(1)).mean()); lo, hi = 1.0, 1000.0
    for _ in range(40):
        mid = (lo * hi) ** .5; lo, hi = (mid, hi) if unc(mid) < .38 else (lo, mid)
T0 = (lo * hi) ** .5; print(f"T0 = {T0:.2f} (inversions' 1 - sum p_T^2 = {unc(T0):.3f}; at T = 1: {unc(1.0):.4f})", flush=True)
L = torch.tensor(np.load("cache/e14_what_e9.npz")["top_L"][0], dtype=torch.float32, device=DEV); P = 801
G = torch.zeros(3, P, P, dtype=torch.float64); r = torch.zeros(3, P, dtype=torch.float64)
with torch.no_grad():
    for i in range(0, 256, 16):
        ins = inputs_3b(m, to_3a(m, prep(INV[i:i + 16]))); Xu = F.unfold(F.pad(ins[2], (2, 2, 2, 2)), 5); Wd = torch.cat([Xu, torch.ones_like(Xu[:, :1])], 1).transpose(1, 2).reshape(-1, P)
        T = F.relu(pre_3b(m, ins[0])[:, fam]).permute(0, 2, 3, 1).reshape(-1, 10) @ L.T
        for j in range(3): o = T[:, j] > 0; Xs = Wd[o]; G[j] += (Xs.T @ Xs).cpu().double(); r[j] += (Xs.T @ T[o, j]).cpu().double()
V = torch.stack([torch.linalg.solve(G[j] + 1e-7 * torch.trace(G[j]) / P * torch.eye(P, dtype=torch.float64), r[j]) for j in range(3)]).float().to(DEV)
S = Family(m, fam, 3).to(DEV); S.set_weights(V[:, :800].reshape(3, 32, 5, 5), V[:, 800], [c.to(DEV) for c in M.weights()[2]])
def plain_beta(fm, cols):
    Sc, Sz, n = [0, 0], [0, 0], [0, 0]
    with torch.no_grad():
        for i in range(0, 256, 50):
            ins = inputs_3b(m, to_3a(m, prep(INV[i:i + 50]))); _, z1, z2 = intact(m, fam, ins); c1, c2 = coarse(fm, ins)
            for j, (z, c) in enumerate([(z1, c1), (z2, c2)]): Sz[j] = Sz[j] + z.sum((0, 2, 3)); Sc[j] = Sc[j] + c.sum((0, 2, 3)); n[j] += z.shape[0] * z.shape[2] * z.shape[3]
    return [A[j] @ (Sz[min(j // 3, 1)] / n[min(j // 3, 1)]) - cols[j] @ (Sc[min(j // 3, 1)] / n[min(j // 3, 1)]) for j in range(4)]
orig = W.upper; B = {}
for tag, fm in [("M read", M), ("self read", S)]:
    ls, _ = lsq_write(m, fam, INVS, fm, np.arange(256), 3); B[f"{tag}: lsq(inv)"] = (fm, ls, plain_beta(fm, ls))
    for T in [T0 / 2, T0, 2 * T0]:
        W.upper = lambda mm, outs, T=T: orig(mm, outs) / T
        c0, _, _ = W.solve(m, fam, SET, fm, np.arange(len(SET.X)), 10, 3); W.upper = orig
        B[f"{tag}: Fisher(inv, T={T:.1f}) + plain beta"] = (fm, c0, plain_beta(fm, c0))
    print("done", tag, flush=True)
K = real_kl(m, fam, D, D.dv, B)
for k, v in K.items():
    base = K[k.split(":")[0] + ": lsq(inv)"]; d = v - base
    print(f"  {k:40s} dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f}) | vs lsq(inv) {d.mean():+.4f} ({d.std() / np.sqrt(len(d)):.4f})")
np.savez("cache/e20_temp.npz", **{k.replace(' ', '_').replace(':', ''): v for k, v in K.items()})
