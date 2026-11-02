"""E8: P2 with network-made inputs only. Read: Lemma 14.1 (arcs #2) on the class inversions. Write: exact step1 Fisher
on the boundary inputs (+ mirror images). usage: e8_p2.py <boundary npy>"""
import sys, numpy as np, torch
from inception import *
from family import Data, Family
from w_fisher import load_build, solve, real_kl
from u4_w3 import solve_active_k, lsq_write
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); M = load_build(m, fam, "L-k3-m-lr3e-2")
for p in m.parameters(): p.requires_grad_(False)
Bx = np.load(sys.argv[1]); Bx = np.concatenate([Bx, Bx[:, :, ::-1]]); INV = np.load("cache/e1_self_256.npy")
class Src:
    def __init__(self, X): self.X = X
BND, SELF = Src(Bx), Src(INV)
# self read: Lemma 14.1 for arcs #2 on the class inversions (as in e2_compare.py)
WHAT = sys.argv[2] if len(sys.argv) > 2 else "cache/u5_what.npz"; WI = 0 if "e9" in WHAT else 1
L = torch.tensor(np.load(WHAT)["top_L"][WI], dtype=torch.float32, device=DEV); P = 801; print("WHAT from", WHAT, WI)
G = torch.zeros(3, P, P, dtype=torch.float64); r = torch.zeros(3, P, dtype=torch.float64)
with torch.no_grad():
    for i in range(0, len(INV), 16):
        ins = inputs_3b(m, to_3a(m, prep(INV[i:i + 16]))); Xu = F.unfold(F.pad(ins[2], (2, 2, 2, 2)), 5)
        W = torch.cat([Xu, torch.ones_like(Xu[:, :1])], 1).transpose(1, 2).reshape(-1, P)
        T = F.relu(pre_3b(m, ins[0])[:, fam]).permute(0, 2, 3, 1).reshape(-1, 10) @ L.T
        for j in range(3):
            o = T[:, j] > 0; Xs = W[o]; G[j] += (Xs.T @ Xs).cpu().double(); r[j] += (Xs.T @ T[o, j]).cpu().double()
V = torch.stack([torch.linalg.solve(G[j] + 1e-7 * torch.trace(G[j]) / P * torch.eye(P, dtype=torch.float64), r[j]) for j in range(3)]).float().to(DEV)
S = Family(m, fam, 3).to(DEV); S.set_weights(V[:, :800].reshape(3, 32, 5, 5), V[:, 800], [c.to(DEV) for c in M.weights()[2]])
idx = np.arange(len(Bx)); B = {}
for tag, fm in [("M read", M), ("self read", S)]:
    c0, b0, _ = solve(m, fam, BND, fm, idx, 10, 3); c1, b1, woke = solve_active_k(m, fam, BND, fm, idx, (c0, b0), 3)
    B[f"{tag} + boundary step1"] = (fm, c1, b1); print(f"{tag}: woken {woke:.4f}", flush=True)
B["self read + self lsq (E4 ref)"] = (S, *lsq_write(m, fam, SELF, S, np.arange(len(INV)), 3))
B["M read + lsq on boundary"] = (M, *lsq_write(m, fam, BND, M, idx, 3))
K = real_kl(m, fam, D, D.dv, B)
for k, v in K.items(): print(f"  {k:30s} dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})")
ref = np.load("cache/u9_kl_35000.npz")
for a, b in [("M read + boundary step1", "M read + lsq on boundary"), ("self read + boundary step1", "self read + self lsq (E4 ref)")]:
    d = K[a] - K[b]; print(f"  paired {a} - {b}: {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})")
d = K["self read + boundary step1"] - ref["M_read_+_trained_write"]; print(f"  DATA-FREE FORMULA vs bar B: {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})")
d = K["self read + boundary step1"] - ref["arcs_read_+_step1"]; print(f"  DATA-FREE FORMULA vs THE FORMULA (FITTED, 35000 real): {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})")
np.savez("cache/e8_p2_" + ("e9" if "e9" in WHAT else "u5") + ".npz", **{k.replace(' ', '_'): v for k, v in K.items()})
