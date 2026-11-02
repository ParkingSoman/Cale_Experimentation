"""Is the coarse kernel built from SHIFTED fine curve kernels? Basis: each fine 5x5 kernel shifted by (dy,dx) in
{-2..2}^2 inside the 5x5 window (cropped). Share of the coarse kernel's energy in that span, vs nulls: same basis from
10 random other 5x5-branch neurons, and random directions. Then: which shifts carry the coarse kernel (per neuron)."""
import sys, numpy as np, torch
from inception import *
from family import Family
m = load_model(); S1 = np.load("cache/s1_fine.npz"); fam = list(S1["fam"])
ck = torch.load(f"cache/ckpt/{sys.argv[1]}.pt", map_location="cpu"); fm = Family(m, ck["fam"], ck["k"]); fm.load_state_dict(ck["state"])
Rk = fm.weights()[0][0].numpy()                                  # (32,5,5)
W5 = m.mixed3b_5x5_pre_relu_conv.weight.detach().cpu().numpy()   # (96,32,5,5)
def shifted(K):
    out = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            S = np.zeros_like(K)
            ys, yd = (slice(max(0, dy), 5 + min(0, dy)), slice(max(0, -dy), 5 + min(0, -dy)))
            xs, xd = (slice(max(0, dx), 5 + min(0, dx)), slice(max(0, -dx), 5 + min(0, -dx)))
            S[:, ys, xs] = K[:, yd, xd]; out.append(S.ravel())
    return np.array(out)
def share(B, v):
    c, *_ = np.linalg.lstsq(B.T, v, rcond=None); return 1 - ((v - B.T @ c)**2).sum() / (v**2).sum(), c
Bc = np.concatenate([shifted(W5[n - 320]) for n in fam]); v = Rk.ravel()
s, c = share(Bc, v); print(f"rank of shifted-curve basis {np.linalg.matrix_rank(Bc)} of 800; coarse share {s:.3f}")
rng = np.random.default_rng(0); others = [i for i in range(320, 416) if i not in fam]
nul = [share(np.concatenate([shifted(W5[n - 320]) for n in rng.choice(others, 10, replace=False)]), v)[0] for _ in range(10)]
rnd = [share(Bc, rng.standard_normal(800))[0] for _ in range(10)]
print(f"null: shifted basis of 10 random other neurons {np.mean(nul):.3f} (min {min(nul):.3f} max {max(nul):.3f}); random direction in curve basis {np.mean(rnd):.3f}")
print(f"unshifted curve span (10 dims): {share(np.array([W5[n-320].ravel() for n in fam]), v)[0]:.3f}")
# where does each fine neuron's contribution sit? energy of coefficient per shift, per neuron (orientation order)
C = c.reshape(10, 5, 5); np.set_printoptions(precision=2, suppress=True, linewidth=150)
pref = {330: 10, 340: 50, 324: 70, 388: 120, 343: 170, 342: 180, 385: 250, 406: 290, 379: 320, 349: 350}
for i, n in enumerate(fam):
    k = np.unravel_index(np.abs(C[i]).argmax(), (5, 5)); dy, dx = k[0] - 2, k[1] - 2
    print(f"neuron {n} (pref {pref[n]:3d} deg): biggest coefficient {C[i][k]:+.2f} at shift (dy {dy:+d}, dx {dx:+d});  coeff map:\n{C[i]}")
