# serves: H-contour
"""Part C estimator for H-contour (COARSE-THEORY.md, Part C "Estimators"): growth exponent of
Dhat(s) = weighted mean of (phi - 2 y'/s')^2 over curve-member pairs, weights = positive part of the covariance.

Usage: pc_hcontour.py N_IMAGES   (64 = smoke run, 2000 = the run). One forward pass to 3b; the rest is CPU.
"""
import sys
import numpy as np
import torch
from inception import load_model, load_images, split, prep, to_3a, pre_3b

n_img = int(sys.argv[1])
FAM = [330, 340, 324, 388, 343, 342, 385, 406, 379, 349]
TH = np.radians([10, 50, 70, 120, 170, 180, 250, 290, 320, 350.0])
R = 6                                   # max offset in cells
m = load_model()
X, y = load_images()
tr, _, _ = split(y)
idx = tr[:n_img]
acts = []
with torch.no_grad():
    for k in range(0, n_img, 32):
        a3 = to_3a(m, prep(X[idx[k:k + 32]]))
        acts.append(torch.relu(pre_3b(m, a3))[:, FAM].cpu().numpy())
H = np.concatenate(acts).astype(np.float64)                       # (n, 10, 28, 28)
mu = H.mean(axis=(0, 2, 3))
Hc = H - mu[None, :, None, None]
n, N, S, _ = H.shape


def cov(dr, dc):
    """Cov_ij(v) for v = (dr rows, dc cols): E[h_i(p) h_j(p + v)] - mu_i mu_j over the valid overlap."""
    r0, r1 = max(0, -dr), min(S, S - dr)
    c0, c1 = max(0, -dc), min(S, S - dc)
    A = Hc[:, :, r0:r1, c0:c1]
    B = Hc[:, :, r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    return np.einsum("nirc,njrc->ij", A, B) / (A.shape[0] * A.shape[2] * A.shape[3])


offs = [(dr, dc) for dr in range(-R, R + 1) for dc in range(-R, R + 1) if (dr, dc) != (0, 0) and dr * dr + dc * dc <= R * R]
CV = {v: cov(*v) for v in offs}

# tangent axis of each member: principal axis of its autocovariance (positive part), x = col, y = -row
a = np.zeros(N)
ratio_i = []
for i in range(N):
    M2 = np.zeros((2, 2))
    for (dr, dc), C in CV.items():
        w = max(C[i, i], 0.0)
        vx, vy = dc, -dr
        M2 += w * np.array([[vx * vx, vx * vy], [vx * vy, vy * vy]])
    ev, EV = np.linalg.eigh(M2)
    a[i] = np.arctan2(EV[1, 1], EV[0, 1]) % np.pi
    ratio_i.append(ev[0] / ev[1])
off = np.angle(np.exp(2j * (a - TH))) / 2                         # constant offset mod pi, per member
print("reason check: autocovariance eigenvalue ratio min/max per member:", np.round(ratio_i, 2))
print("tangent axes a_i (deg):", np.round(np.degrees(a)).astype(int))
print(f"diagnostic: a_i - theta_i mod 180 (deg): {np.round(np.degrees(off)).astype(int)}  "
      f"circular spread {np.degrees(np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(2j * off)))) ) / 2):.0f}")


def stat(theta):
    rows = []
    for (dr, dc), C in CV.items():
        vx, vy = dc, -dr
        for i in range(N):
            ca, sa = np.cos(a[i]), np.sin(a[i])
            s, yy = ca * vx + sa * vy, -sa * vx + ca * vy
            if abs(s) < 1e-9:
                continue
            sp, yp = abs(s), yy * np.sign(s)
            if abs(yp) > sp:
                continue
            for j in range(N):
                w = max(C[i, j], 0.0)
                if w > 0:
                    phi = np.angle(np.exp(1j * (theta[j] - theta[i])))
                    rows.append((sp, yp, phi, w))
    r = np.array(rows)
    bins = np.arange(1, R + 1)
    D, beta, W = [], [], []
    for bcell in bins:
        sel = np.abs(r[:, 0] - bcell) < 0.5
        sp, yp, phi, w = r[sel].T
        D.append(np.sum(w * (phi - 2 * yp / sp) ** 2) / np.sum(w))
        u = yp / sp
        beta.append(np.sum(w * u * phi) / np.sum(w * u * u))
        W.append(w.sum())
    return bins.astype(float), np.array(D), np.array(beta), np.array(W)


def fit(sv, D, p):
    from scipy.optimize import nnls
    A = np.stack([np.ones_like(sv), sv ** p, 1 / sv ** 2], 1)
    coef, res = nnls(A, D)
    return coef, res


sv, D, beta, W = stat(TH)
print("s (cells):", sv.astype(int))
print("Dhat(s):  ", np.round(D, 4))
print("slope beta of phi on y'/s' per bin:", np.round(beta, 2))
print("total weight per bin:", np.array2string(W, precision=3))
for p in (3, 1):
    coef, res = fit(sv, D, p)
    print(f"fit p={p}: a={coef[0]:.4f} b={coef[1]:.2e} c={coef[2]:.4f}  residual norm {res:.2e}")
r3, r1 = fit(sv, D, 3)[1], fit(sv, D, 1)[1]
ratio = max(r3, r1) / max(min(r3, r1), 1e-15)
verdict = "undecided" if ratio <= 2 else ("p = 3 (curvature process)" if r3 < r1 else "p = 1 (direction process)")
print(f"residual ratio {ratio:.2f} -> {verdict}")
rng = np.random.default_rng(0)
for t in range(3):
    svn, Dn, bn, _ = stat(rng.permutation(TH))
    print(f"null (shuffled angles) {t}: Dhat {np.round(Dn, 3)}  slopes {np.round(bn, 2)}")
