# serves: TQ9
"""Certified always-on least with one coarse unit (Lemma TQ9, COARSE-THEORY.md B.Q), n_y = 4, two areas, on
(A) a hand-built instance and (B) a small ReLU network TRAINED by gradient descent (numpy) on a task with two groups of
classes; for (B) the areas are the two input clusters and the features are the hidden pre-activations (centred per
area). Branch and bound over the hemisphere of unit v in the image of P_perp (coordinates (phi, psi), cell radius
theta = (dphi + dpsi)/2, a safe bound on the chord): each cell gets the primal value F(v_0) (an upper value), an exactly
feasible dual point (Lemma TQ9 part 1: q from the primal softmax, corrected to meet the moment constraints by a
least-norm step) giving D(v_0) <= F(v_0), and the code bound of part 2 (pairs of points, two-class KL). Cells whose
lower value exceeds the best upper value are dropped; the rest are split. Output: certified lower bound LB, best upper
UB, F at the shared span (top eigenvector of the summed area covariances of P_perp y) and F at the TQ10 span (the
shared span tilted by Theorem TQ10's least squares, k = 1). Cell step: the DUAL step of Lemma TQ9 part 4 (the first
version's code-bound step pruned nothing). Prune tolerance 5% of F(shared). If the cell cap is reached, the reported LB is partial: the least lower value
over the cells still open (still a proved lower bound). Fails: a primal value below the certified LB (impossible if the certificate is sound)."""
import json
import os

import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import logsumexp, expit

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20261102)
NY = 4
E = np.linalg.qr(np.eye(NY)[:, :-1] - 1.0 / NY)[0]          # orthonormal basis of the image of P_perp (4 x 3)


def hand_built():
    areas = []
    for w, (blk, oth) in enumerate([([0, 1], [2, 3]), ([2, 3], [0, 1])]):
        Z = rng.normal(size=(12, 2)); Z -= Z.mean(0)
        B = np.zeros((NY, 2)); B[blk[0], 0] = 2.0; B[blk[1], 0] = -2.0; B[oth[0], 1] = 0.6; B[blk[0], 1] = 0.8
        base = np.zeros(NY); base[blk] = 2.5
        Y = base + Z @ B.T
        areas.append(dict(Z=Z, y=Y, p=np.full(12, 1 / 24)))
    return areas


def trained():
    """1 hidden layer (8 ReLU), input 3; cluster 0 -> classes {0,1} by x1, cluster 1 -> classes {2,3} by x2."""
    n = 400
    cl = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 3)) * 0.8; X[:, 0] += np.where(cl == 0, -2.0, 2.0)
    lab = np.where(cl == 0, (X[:, 1] > 0).astype(int), 2 + (X[:, 2] > 0).astype(int))
    flip = rng.random(n) < 0.1; lab[flip] = rng.integers(0, NY, flip.sum())
    W1 = rng.normal(size=(8, 3)) * 0.5; b1 = np.zeros(8); W2 = rng.normal(size=(NY, 8)) * 0.5; b2 = np.zeros(NY)
    Yoh = np.eye(NY)[lab]
    for it in range(4000):
        Zp = X @ W1.T + b1; H = np.maximum(Zp, 0); L = H @ W2.T + b2
        Q = np.exp(L - logsumexp(L, axis=1, keepdims=True))
        G = (Q - Yoh) / n
        gW2 = G.T @ H; gb2 = G.sum(0); GH = (G @ W2) * (Zp > 0); gW1 = GH.T @ X; gb1 = GH.sum(0)
        lr = 0.5
        W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
    areas = []
    for c in [0, 1]:
        Xc = rng.normal(size=(12, 3)) * 0.8; Xc[:, 0] += (-2.0 if c == 0 else 2.0)
        Zp = Xc @ W1.T + b1; y = np.maximum(Zp, 0) @ W2.T + b2
        areas.append(dict(Z=Zp - Zp.mean(0), y=y, p=np.full(12, 1 / 24)))
    acc = float(np.mean(np.argmax(np.maximum(X @ W1.T + b1, 0) @ W2.T + b2, 1) == lab))
    return areas, acc


def prep(areas):
    for a in areas:
        a["ls"] = a["y"] - logsumexp(a["y"], axis=1, keepdims=True); a["s"] = np.exp(a["ls"])
    return areas


def fit_area(a, v, x0=None):
    Z, s, ls, p = a["Z"], a["s"], a["ls"], a["p"]; d = Z.shape[1]
    def f(th):
        aa, t = th[:d], th[d:]
        c = Z @ aa; L = t[None, :] + np.outer(c, v)
        lq = L - logsumexp(L, axis=1, keepdims=True); q = np.exp(lq)
        val = float(np.sum(p[:, None] * s * (ls - lq)))
        G = p[:, None] * (q - s)
        return val, np.concatenate([Z.T @ (G @ v), G.sum(0)])
    th0 = np.zeros(d + NY) if x0 is None else x0
    th0[d:] = a["ls"].mean(0) if x0 is None else th0[d:]
    r = minimize(f, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=3000, gtol=1e-12, ftol=1e-15))
    aa, t = r.x[:d], r.x[d:]
    c = Z @ aa; L = t[None, :] + np.outer(c, v); q = np.exp(L - logsumexp(L, axis=1, keepdims=True))
    return r.fun, q, c, r.x


def dual_value(a, v, q):
    """correct q to exact feasibility (least-norm), return sum p (H(q) - H(s)) or None if it leaves the simplex."""
    Z, s, p = a["Z"], a["s"], a["p"]; nw, d = Z.shape
    rows = []; rhs = []
    for i in range(NY):                                     # sum_x p_x (q_x - s_x)_i = 0
        r = np.zeros((nw, NY)); r[:, i] = p; rows.append(r.ravel()); rhs.append(-np.sum(p * (q[:, i] - s[:, i])))
    for j in range(d):                                      # sum_x p_x (v^T (q_x - s_x)) z_xj = 0
        r = (p * Z[:, j])[:, None] * v[None, :]; rows.append(r.ravel())
        rhs.append(-np.sum(p * Z[:, j] * ((q - s) @ v)))
    for x in range(nw):                                     # each correction sums to 0
        r = np.zeros((nw, NY)); r[x, :] = 1; rows.append(r.ravel()); rhs.append(0.0)
    A = np.array(rows); b = np.array(rhs)
    dl = np.linalg.lstsq(A, b, rcond=None)[0]
    if np.abs(A @ dl - b).max() > 1e-11:
        return None
    qq = q + dl.reshape(nw, NY)
    if qq.min() <= 0:
        return None
    H = lambda P: -np.sum(P * np.log(P), axis=1)
    return float(np.sum(p * (H(qq) - H(s)))), qq, A


def dual_step(a, v0, qq, A, theta, tau=0.05):
    """Lemma TQ9 part 4, head form: the correction moves only coordinates with q* >= tau (the head)."""
    Z, s, p = a["Z"], a["s"], a["p"]
    mask = (qq >= tau).ravel()
    if np.any((qq >= tau).sum(1) < 2):
        return np.inf
    Ah = A[:, mask]
    keep = np.abs(Ah).sum(1) > 0                        # rows that vanish on the head have right side 0 (first block)
    first = [i for i in range(NY) if keep[i]]
    if first:
        keep[first[0]] = False                          # drop one first-block row: the rows satisfy one relation
    Ah = Ah[keep]
    sv = np.linalg.svd(Ah, compute_uv=False)
    if sv[-1] <= 1e-10 * sv[0] or len(sv) < Ah.shape[0]:
        return np.inf                                   # Lemma TQ9 head form: full row rank at v_0 is required
    sigma0 = sv[-1]
    mstar = float(np.sum(p * np.linalg.norm(qq - s, axis=1) * np.linalg.norm(Z, axis=1)))
    sigma = sigma0 - theta * np.sqrt(np.sum(p ** 2 * np.sum(Z ** 2, axis=1)))
    qmin = qq[qq >= tau].min()
    if sigma <= 0:
        return np.inf
    delta = theta * mstar / sigma
    if delta >= qmin:
        return np.inf
    return float(np.sqrt(NY) * (1 + np.log(1 / qmin)) * np.sqrt(np.sum(p ** 2)) * delta
                 + p.max() * delta ** 2 / (2 * (qmin - delta)))


def kl2_dev(alpha, m):
    """least KL(sigma(alpha) || sigma(alpha + mu)) over |mu| >= m."""
    if m <= 0:
        return 0.0
    sa = expit(alpha)
    def k(mu):
        sb = expit(alpha + mu)
        return sa * np.log(sa / sb) + (1 - sa) * np.log((1 - sa) / (1 - sb))
    return min(k(m), k(-m))


def code_bound(a, v0, theta, F0):
    s, p = a["s"], a["p"]; nw = len(p)
    i, j = int(np.argmax(v0)), int(np.argmin(v0))
    gap = v0[i] - v0[j] - 2 * theta
    if gap <= 0:
        return np.inf
    lr = np.log(s[:, i] / s[:, j]); mass = s[:, i] + s[:, j]
    R = np.zeros(nw)
    for x in range(nw):
        best = 0.0
        for x2 in range(nw):
            if x2 == x:
                continue
            off = abs(lr[x] - lr[x2])
            def bound(delta):
                m = (delta * gap - off) / 2
                b1 = p[x] * mass[x] * kl2_dev(lr[x], m); b2 = p[x2] * mass[x2] * kl2_dev(lr[x2], m)
                return min(b1, b2) - F0
            hi = 1.0
            while bound(hi) < 0 and hi < 1e6:
                hi *= 2
            if hi >= 1e6:
                return np.inf
            D = brentq(bound, 0.0, hi) if bound(0.0) < 0 else 0.0
            best = max(best, D)
        R[x] = best
    return float(np.sum(p * R))


def uvec(phi, psi):
    u = np.array([np.sin(phi) * np.cos(psi), np.sin(phi) * np.sin(psi), np.cos(phi)])
    return E @ u


def certify(areas, tol, max_cells=150000):
    def F_and_D(v):
        Fv = 0.0; Dv = 0.0; fams = []
        for a in areas:
            f, q, c, _ = fit_area(a, v)
            r = dual_value(a, v, q)
            Fv += f
            if r is None:
                Dv = -np.inf; fams.append(None)
            else:
                Dv += r[0]; fams.append((r[1], r[2]))
        return Fv, Dv, fams
    queue = []
    for i in range(6):
        for j in range(16):
            queue.append((i * np.pi / 12, (i + 1) * np.pi / 12, j * np.pi / 8, (j + 1) * np.pi / 8))
    UB = np.inf; LB_dropped = np.inf; n_eval = 0; rnd = 0
    while queue and n_eval < max_cells:
        rnd += 1; new = []; evals = []; min_open = np.inf
        for (p0, p1, s0, s1) in queue:
            v0 = uvec((p0 + p1) / 2, (s0 + s1) / 2)
            Fv, Dv, fams = F_and_D(v0); n_eval += 1
            UB = min(UB, Fv); evals.append(((p0, p1, s0, s1), v0, Dv, fams))
        for (cell, v0, Dv, fams) in evals:
            p0, p1, s0, s1 = cell
            theta = ((p1 - p0) + (s1 - s0)) / 2
            if any(f is None for f in fams) or not np.isfinite(Dv):
                lb = -np.inf
            else:
                lb = Dv - sum(dual_step(a, v0, f[0], f[1], theta) for a, f in zip(areas, fams))
            if lb >= UB - tol:
                LB_dropped = min(LB_dropped, max(lb, UB - tol)); continue
            min_open = min(min_open, lb)
            pm, sm = (p0 + p1) / 2, (s0 + s1) / 2
            new += [(p0, pm, s0, sm), (p0, pm, sm, s1), (pm, p1, s0, sm), (pm, p1, sm, s1)]
        queue = new
        print("  round %d: cells left %d, UB %.6f, evals %d, partial LB %.6f" % (rnd, len(queue), UB, n_eval,
              min(LB_dropped, min_open)), flush=True)
    LB = min(LB_dropped, UB) if not queue else min(LB_dropped, min_open)
    return LB, UB, n_eval, len(queue)


def lin_read(a):
    """least-squares linear read of the centred logits on the features (eps_lin of DP1 is its residual)."""
    Pp = a["y"] - a["y"].mean(1, keepdims=True); Yc = Pp - (a["p"][:, None] * Pp).sum(0) / a["p"].sum()
    B = np.linalg.lstsq(a["Z"], Yc, rcond=None)[0].T
    return B, Yc


def shared_and_tq10(areas):
    Sig = np.zeros((NY, NY)); Bs = []
    for a in areas:
        B, Yc = lin_read(a); Bs.append(B)
        Sig += (a["p"][:, None] * (a["Z"] @ B.T)).T @ (a["Z"] @ B.T)
    ev, U = np.linalg.eigh(Sig); q0 = U[:, -1]
    Fsh = sum(fit_area(a, q0)[0] for a in areas)
    # TQ10 leading-order tilt (k = 1): least squares for Delta (ny), m_w (1 x d), beta_w, per-point Fisher metric at e = 0
    rows = []; rhs = []; nvar = NY; offs = []
    for a in areas:
        offs.append(nvar); nvar += a["Z"].shape[1] + NY
    for wi, (a, B) in enumerate(zip(areas, Bs)):
        A = q0 @ B; T = B - np.outer(q0, A); o = offs[wi]; d = a["Z"].shape[1]
        ybar = (a["p"][:, None] * a["y"]).sum(0) / a["p"].sum()
        for xi in range(len(a["p"])):
            z = a["Z"][xi]; y0 = ybar + q0 * (A @ z); s0 = np.exp(y0 - logsumexp(y0))
            Hh = np.diag(s0) - np.outer(s0, s0); ev2, U2 = np.linalg.eigh(Hh)
            L = (U2 * np.sqrt(np.clip(ev2, 0, None))) @ U2.T * np.sqrt(a["p"][xi])
            G = np.zeros((NY, nvar)); G[:, :NY] = np.eye(NY) * (A @ z)
            G[:, o:o + d] = np.outer(q0, z); G[:, o + d:o + d + NY] = np.eye(NY)
            rows.append(L @ G); rhs.append(L @ (T @ z))
    sol = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
    v = q0 + sol[:NY]; v = v - v.mean(); v /= np.linalg.norm(v)
    Ftq = sum(fit_area(a, v)[0] for a in areas)
    return Fsh, Ftq


out = {}
for name in ["hand_built", "trained"]:
    if name == "hand_built":
        areas = prep(hand_built()); extra = {}
    else:
        ar, acc = trained(); areas = prep(ar); extra = dict(train_accuracy=acc)
    print(name, "start", extra, flush=True)
    Fsh, Ftq = shared_and_tq10(areas)
    LB, UB, ne, left = certify(areas, tol=0.05 * Fsh, max_cells=200000)
    r = dict(shared=Fsh, tq10_rule=Ftq, certified_LB=LB, best_UB=UB, shared_over_LB=Fsh / LB if LB > 0 else None,
             tq10_over_LB=Ftq / LB if LB > 0 else None, cells=ne, cells_left=left, **extra)
    print(name, "RESULT", r, flush=True)
    out[name] = r
json.dump(out, open(os.path.join(HERE, "tq_cert.json"), "w"), indent=1)
