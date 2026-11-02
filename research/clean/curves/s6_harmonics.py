"""Reader weights on the curve family, decomposed into harmonics over preferred orientation (least squares on
cos/sin basis at the measured, uneven orientations). Energy share per harmonic, per 4a input conv. Null: the same
neurons with orientations shuffled (100 shuffles)."""
import numpy as np, torch
from inception import *
from family import IN4A
m = load_model(); S1 = np.load("cache/s1_fine.npz"); fam = list(S1["fam"])
pref = np.deg2rad(np.array([10, 50, 70, 120, 170, 180, 250, 290, 320, 350]))   # orientation order (S1)
def basis(th, H=4):
    cols = [np.ones_like(th)]
    for h in range(1, H + 1): cols += [np.cos(h * th), np.sin(h * th)]
    return np.stack(cols, 1)                                     # (10, 1+2H)
def shares(A, th, H=4):
    B = basis(th, H); Q, _ = np.linalg.qr(B)                     # orthonormal over the 10 points
    P = A @ Q                                                    # rows x (1+2H)
    e = (P**2).sum(0); tot = (A**2).sum()
    return np.array([e[0]] + [e[2*h-1] + e[2*h] for h in range(1, H + 1)]) / tot
rng = np.random.default_rng(0); np.set_printoptions(precision=3, suppress=True)
for n in IN4A + ["all"]:
    A = (np.concatenate([getattr(m, k).weight.detach().cpu()[:, fam, 0, 0].numpy() for k in IN4A]) if n == "all"
         else getattr(m, n).weight.detach().cpu()[:, fam, 0, 0].numpy())
    s = shares(A, pref); nul = np.mean([shares(A, rng.permutation(pref)) for _ in range(100)], 0)
    print(f"{n:38s} harmonic 0,1,2,3,4 share: {s}   shuffled-orientation null: {nul}")
# same for the coarse-relevant input side: the fine kernels themselves (800-dim rows -> columns are neurons)
W = S1["W"]                                                       # (10, 800) orientation order
s = shares(W.T, pref); nul = np.mean([shares(W.T, rng.permutation(pref)) for _ in range(100)], 0)
print(f"{'fine kernels (input side)':38s} harmonic share: {s}   null: {nul}")
