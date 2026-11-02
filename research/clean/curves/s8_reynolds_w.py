"""Weight space: K_R(r) = sum_theta shift(W_theta, toward centre of curvature by r positions), cropped to 5x5.
cos with the trained coarse kernel; controls: reversed direction, random orientation labels."""
import sys, numpy as np, torch
from inception import *
from family import Family
m = load_model(); S1 = np.load("cache/s1_fine.npz"); fam = list(S1["fam"]); W = S1["W"].reshape(10, 32, 5, 5)
pref = np.deg2rad(np.array([10, 50, 70, 120, 170, 180, 250, 290, 320, 350]))
ck = torch.load(f"cache/ckpt/{sys.argv[1]}.pt", map_location="cpu"); fm = Family(m, ck["fam"], ck["k"]); fm.load_state_dict(ck["state"])
K = fm.weights()[0][0].numpy()
def sh(k, dy, dx):   # kernel for z(q) = z_theta(q + (dy,dx)) : out[:, a, b] = k[:, a - dy, b - dx]
    o = np.zeros_like(k)
    for a in range(5):
        for b in range(5):
            if 0 <= a - dy < 5 and 0 <= b - dx < 5: o[:, a, b] = k[:, a - dy, b - dx]
    return o
def KR(r, th):
    return sum(sh(W[i], int(round(r * np.sin(t))), int(round(-r * np.cos(t)))) for i, t in enumerate(th))
cos = lambda a, b: (a * b).sum() / np.linalg.norm(a) / np.linalg.norm(b)
rng = np.random.default_rng(0)
for r in [0, 1, 2, 3]:
    nul = [cos(KR(r, rng.permutation(pref)), K) for _ in range(200)]
    print(f"r={r}: cos(K_R, coarse) {cos(KR(r, pref), K):+.3f}  reversed {cos(KR(-r, pref), K):+.3f}  shuffled-orientation null {np.mean(nul):+.3f} (sd {np.std(nul):.3f})")
