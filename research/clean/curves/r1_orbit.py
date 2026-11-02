"""R1 (NEURON-MATH section 4): is the curve family one rotation orbit? Weights only, no images.
(a) norm CV, (b) even circulant fit of the kernel and column Grams vs angle shuffles,
(c) angle recovery by channel-free Procrustes (Lemma 4.2(3)), (d) exact 90/180-degree grid pairs."""
import numpy as np, torch
from scipy.ndimage import rotate
from inception import load_model
from family import IN4A

m = load_model()
FAM = [330, 340, 324, 388, 343, 342, 385, 406, 379, 349]
TH = np.array([10, 50, 70, 120, 170, 180, 250, 290, 320, 350], float)
W5 = m.mixed3b_5x5_pre_relu_conv.weight.detach().cpu().numpy()          # (96, 32, 5, 5)
K = W5[[i - 320 for i in FAM]]
OTHER = [j for j in range(96) if j + 320 not in FAM]
COL = np.concatenate([getattr(m, n).weight.detach().cpu().numpy()[:, FAM, 0, 0] for n in IN4A], 0)   # (368, 10)
rng = np.random.default_rng(0)

# (a)
nk, nw = np.linalg.norm(K.reshape(10, -1), axis=1), np.linalg.norm(COL, axis=0)
print(f"(a) CV |K| {nk.std() / nk.mean():.3f}   CV |W| {nw.std() / nw.mean():.3f}")

# (b)
pairs = [(i, j) for i in range(10) for j in range(i + 1, 10)]
def circ_r2(G, th):
    x = np.deg2rad(np.array([th[j] - th[i] for i, j in pairs]))
    y = np.array([G[i, j] for i, j in pairs])
    A = np.stack([np.ones_like(x), np.cos(x), np.cos(2 * x)], 1)
    r = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return 1 - r.var() / y.var()
for name, M in [("K", K.reshape(10, -1)), ("W", COL.T)]:
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True); G = Mn @ Mn.T
    r2 = circ_r2(G, TH); null = np.array([circ_r2(G, rng.permutation(TH)) for _ in range(1000)])
    print(f"(b) {name}: R2 {r2:.3f}   shuffle 95th pct {np.percentile(null, 95):.3f}   rank {np.mean(null < r2):.3f}")

# (c), (d)
def rot(k, a):
    return np.stack([rotate(c, a, reshape=False, order=1, mode="constant") for c in k])
def proc(A, B):          # channel-free residual, unit-normalised: 2 - 2 * nuclear(B A^T)
    A = A.reshape(32, -1); B = B.reshape(32, -1)
    A = A / np.linalg.norm(A); B = B / np.linalg.norm(B)
    return 2 - 2 * np.linalg.svd(B @ A.T, compute_uv=False).sum()
grid = np.arange(0, 360, 5)
for sign in (+1, -1):     # the angle convention (one global bit) is unknown: report both
    errs = []
    for i in range(10):
        for j in range(10):
            if i == j: continue
            res = [proc(rot(K[i], sign * a), K[j]) for a in grid]
            a_star = grid[int(np.argmin(res))]
            d = (a_star - (TH[j] - TH[i])) % 360; errs.append(min(d, 360 - d))
    errs = np.array(errs)
    print(f"(c) sign {sign:+d}: median |alpha* - dtheta| {np.median(errs):.0f} deg, within 45: {np.mean(errs <= 45):.2f}")

for (a, b, ang) in [(50, 320, -90), (70, 250, 180), (170, 350, 180)]:
    i, j = int(np.where(TH == a)[0][0]), int(np.where(TH == b)[0][0])
    for sign in (+1, -1):
        rr = proc(rot(K[i], sign * ang), K[j]) if abs(ang) != 180 else proc(np.rot90(K[i], 2, axes=(1, 2)), K[j])
        others = [proc(np.rot90(K[i], (sign * ang // 90) % 4, axes=(1, 2)), W5[o]) for o in OTHER]
        print(f"(d) {FAM[i]}({a}) -> {FAM[j]}({b}) by {ang}, sign {sign:+d}: rotated {rr:.3f}  unrotated {proc(K[i], K[j]):.3f}"
              f"  other-neuron 5th pct {np.percentile(others, 5):.3f}  min {min(others):.3f}")
        if abs(ang) == 180: break
