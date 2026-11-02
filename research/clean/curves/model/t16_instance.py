# serves: T16
"""Exact instance for Theorem T16 (COARSE-THEORY.md B.6): the optimum over self-contained builds is a search over the
gates' effective reads R, with a closed form J_k(R) inside.

Chain: input field a(x, p) in R^5 at 3 positions, 600 inputs (Gaussian plus a shift), first family K (5 x 4), depth 2,
gates phi_l(t) = softplus(alpha_l t + beta_l) reading one column each (S_1 reads z, S_2 reads v_1), edge w_1 (3 x 3),
head H (6 x 4). Forced reads R*_1 = K S_1, R*_2 = K S_2, so r = 2. Widths k = 2, 3.
Revision after the first run (a script error, not a result): gate 2 reads layer 1, so a build's effective read is
R_2 = K' S'_2 (W'_1 = I); the first run used K' w'_1^T S'_2, which produced false violations of U2 and U5. The sizes
were also raised (d 4 -> 5, N 3 -> 4, n_y 4 -> 6) so that k = 3 is below full width.

Predictions (written before the run):
  U1  J_k(R*) equals the value of Theorem T11 computed independently in member space (about 1e-10 relative).
  U2  1000 random self-contained builds per k: D2(B) >= J_k(R_B) (no violation beyond 1e-10 relative), and the build of
      T16 part 4 made from R_B, run as a network that computes its own gates, has D2 = J_k(R_B) (about 1e-10).
  U3  the gradient of J_k at R* is not zero, and gradient descent on J_k from R* ends below the T11 value.
  U4  multi-start minimisation of J_k over R: the best value is >= the T12 floor and <= the T11 value. Expectation, not
      a theorem: it is <= the best of 4 free-read students trained by Adam on all weights.
  U5  for each trained student, J_k at its effective reads is <= its D2 (part 4).
"""
import json
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
g = torch.Generator().manual_seed(16)
d, N, Pn, n, ny = 5, 4, 3, 600, 6
A = torch.randn(n, Pn, d, generator=g) + 0.3
rho = torch.full((Pn,), 1.0 / Pn)
K = torch.randn(d, N, generator=g)
S1, S2 = torch.randn(N, 1, generator=g), torch.randn(N, 1, generator=g)
w1 = torch.randn(N, N, generator=g) / np.sqrt(N)
H = torch.randn(ny, N, generator=g)
c = torch.randn(ny, generator=g)
al, be = (1.0, 0.8), (0.3, 0.2)
P = torch.eye(ny) - torch.ones(ny, ny) / ny
sp = torch.nn.functional.softplus


def phi(l, t):
    return sp(al[l] * t + be[l])


def gates_from_reads(R1, R2):
    g1 = phi(0, A @ R1)                       # (n, Pn)
    g2 = phi(1, g1 * (A @ R2))
    return g1, g2


# teacher
z = A @ K                                         # (n, Pn, N)
g1t = phi(0, (z @ S1)[..., 0]); v1 = g1t[..., None] * z
g2t = phi(1, (v1 @ S2)[..., 0]); v2 = g2t[..., None] * (v1 @ w1.T)
y = (v2 * rho[None, :, None]).sum(1) @ H.T + c
R1s, R2s = (K @ S1)[:, 0], (K @ S2)[:, 0]
res = {"teacher gates from R* match": float(max((g1t - gates_from_reads(R1s, R2s)[0]).abs().max(),
                                                  (g2t - gates_from_reads(R1s, R2s)[1]).abs().max()))}


def cov(X, Y=None):
    Xc = X - X.mean(0)
    Yc = Xc if Y is None else Y - Y.mean(0)
    return Xc.T @ Yc / X.shape[0]


def sqrtm_sym(M, inv=False):
    e, V = torch.linalg.eigh(M)
    e = e.clamp(min=1e-300)
    return V @ torch.diag(e ** (-0.5 if inv else 0.5)) @ V.T


def J(R1, R2, k, parts=False):
    g1, g2 = gates_from_reads(R1, R2)
    u = ((g1 * g2)[..., None] * A * rho[None, :, None]).sum(1)          # (n, d)
    Sig = cov(u)
    Py = y @ P
    Y = cov(Py, u) @ sqrtm_sym(Sig, inv=True)
    Fm = sqrtm_sym(Sig) @ torch.stack([R1, R2], 1)
    Q, Rr = torch.linalg.qr(Fm)
    rp = int((Rr.diagonal().abs() > 1e-9 * Rr.diagonal().abs().max()).sum())
    Q = Q[:, :rp]
    PiF = Q @ Q.T
    nu = torch.linalg.eigvalsh(Y @ (torch.eye(d) - PiF) @ Y.T).flip(0)
    total = torch.trace(cov(Py))
    Jv = 0.5 * (total - torch.trace(Y @ PiF @ Y.T) - nu[:max(k - rp, 0)].sum())
    if parts:
        return Jv, dict(gate_error=0.5 * (total - (Y ** 2).sum()), u=u, Sig=Sig, Y=Y, Q=Q, rp=rp)
    return Jv


def t11_value(k):
    """Theorem T11 in member space: forced reads R_0 = span{S1, S2}; residual PCA."""
    G = g1t * g2t
    uhat = ((G[..., None] * z) * rho[None, :, None]).sum(1)              # (n, N)
    Chat = cov(uhat)
    A_ = H @ w1
    R0, _ = torch.linalg.qr(torch.cat([S1, S2], 1))
    Cf = R0.T @ Chat @ R0
    Cyf = P @ A_ @ Chat @ R0
    Cy = P @ A_ @ Chat @ A_.T @ P
    Ce = Cy - Cyf @ torch.linalg.solve(Cf, Cyf.T)
    nu = torch.linalg.eigvalsh(Ce).flip(0)
    return float(0.5 * nu[k - 2:].sum())


def build_forward(Kp, w1p, S1p, S2p, Hp, cp):
    zp = A @ Kp
    gb1 = phi(0, (zp @ S1p)[..., 0]); vb1 = gb1[..., None] * zp
    gb2 = phi(1, (vb1 @ S2p)[..., 0]); vb2 = gb2[..., None] * (vb1 @ w1p.T)
    return (vb2 * rho[None, :, None]).sum(1) @ Hp.T + cp


def d2(yb):
    return float(0.5 * (((yb - y) @ P) ** 2).sum(1).mean())


def floor(k):
    mu = torch.linalg.eigvalsh(cov(y @ P)).flip(0)
    return float(0.5 * mu[k:].sum())


def converse_build(R1, R2, k):
    """T16 part 4: K' = basis of the optimal V, identity edges, reads with K' S' = R, optimal head and bias."""
    Jv, pt = J(R1, R2, k, parts=True)
    Sig, Y, Q, rp = pt["Sig"], pt["Y"], pt["Q"], pt["rp"]
    Mres = (torch.eye(d) - Q @ Q.T) @ Y.T @ Y @ (torch.eye(d) - Q @ Q.T)
    e, V = torch.linalg.eigh(Mres)
    W = torch.cat([Q, V[:, e.argsort(descending=True)[:k - rp]]], 1)               # basis of W = Sig^1/2 V
    Kp = sqrtm_sym(Sig, inv=True) @ W                                              # columns span V
    S1p = torch.linalg.lstsq(Kp, R1[:, None]).solution
    S2p = torch.linalg.lstsq(Kp, R2[:, None]).solution
    # best head on the build's own features (gates are those of R): regress P y on [Kp^T u, 1]
    feats = pt["u"] @ Kp
    Fx = torch.cat([feats, torch.ones(n, 1)], 1)
    coef = torch.linalg.lstsq(Fx, y @ P).solution
    Hp, cp = coef[:k].T, coef[k]
    return Kp, torch.eye(k), S1p, S2p, Hp, cp


for k in (2, 3):
    row = {"floor (T12)": floor(k), "T11 value": t11_value(k)}
    Jstar = float(J(R1s, R2s, k))
    row["U1 J(R*)"] = Jstar
    row["U1 rel err"] = abs(Jstar - row["T11 value"]) / row["T11 value"]
    viol, conv_err = 0.0, 0.0
    for t in range(1000):
        Kp = torch.randn(d, k, generator=g); w1p = torch.randn(k, k, generator=g)
        S1p, S2p = torch.randn(k, 1, generator=g), torch.randn(k, 1, generator=g)
        Hp, cp = torch.randn(ny, k, generator=g), torch.randn(ny, generator=g)
        dB = d2(build_forward(Kp, w1p, S1p, S2p, Hp, cp))
        R1b, R2b = (Kp @ S1p)[:, 0], (Kp @ S2p)[:, 0]
        Jb = float(J(R1b, R2b, k))
        viol = max(viol, (Jb - dB) / Jb)
        if t < 100:
            cb = converse_build(R1b, R2b, k)
            conv_err = max(conv_err, abs(d2(build_forward(*cb)) - Jb) / Jb)
    row["U2 max (J - D2)/J over random builds (<= 0 expected)"] = viol
    row["U2 converse build rel err"] = conv_err
    # U3 gradient at R* and descent
    R = torch.stack([R1s, R2s], 1).clone().requires_grad_(True)
    Jv = J(R[:, 0], R[:, 1], k); Jv.backward()
    row["U3 grad norm at R*"] = float(R.grad.norm())
    opt = torch.optim.LBFGS([R], max_iter=500, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); v = J(R[:, 0], R[:, 1], k); v.backward(); return v
    opt.step(closure)
    row["U3 J after descent from R*"] = float(J(R[:, 0], R[:, 1], k).detach())
    # U4 multi-start
    best = float("inf")
    for s in range(30):
        R = torch.randn(d, 2, generator=g).requires_grad_(True)
        opt = torch.optim.LBFGS([R], max_iter=300, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); v = J(R[:, 0], R[:, 1], k); v.backward(); return v
        try:
            opt.step(closure)
            best = min(best, float(J(R[:, 0], R[:, 1], k)))
        except Exception:
            pass
    row["U4 best J over 30 starts"] = best
    # students
    studs, u5 = [], 0.0
    for s in range(4):
        gs = torch.Generator().manual_seed(100 + s)
        prm = [torch.randn(d, k, generator=gs) * 0.5, torch.randn(k, k, generator=gs) / np.sqrt(k),
               torch.randn(k, 1, generator=gs), torch.randn(k, 1, generator=gs), torch.randn(ny, k, generator=gs) * 0.3,
               torch.zeros(ny)]
        prm = [p.requires_grad_(True) for p in prm]
        opt = torch.optim.Adam(prm, 3e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 6000)
        for it in range(6000):
            loss = 0.5 * (((build_forward(*prm) - y) @ P) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            dS = d2(build_forward(*prm))
            Js = float(J((prm[0] @ prm[2])[:, 0], (prm[0] @ prm[3])[:, 0], k))
        studs.append(dS)
        u5 = max(u5, (Js - dS) / dS)
    row["U4 students D2"] = studs
    row["U5 max (J(R_student) - D2)/D2 (<= 0 expected)"] = u5
    res[f"k={k}"] = row
    print(k, json.dumps(row), flush=True)
json.dump(res, open(__file__.replace(".py", ".json"), "w"), indent=1)
