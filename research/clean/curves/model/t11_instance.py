# serves: T11
# serves: T12
# serves: T11a
"""Exact instance of a self-gated chain (COARSE-THEORY.md B.5). Checks Lemma T11a, Theorem T11, Corollaries T11.1 and
T11.2 and Theorem T12 to machine precision.

The theory uses the input only through the first family's pre-activations z(x, p) = K^T a(x, p). So the instance
gives z directly: a(x, p) is the indicator of the pair (x, p) and K is the table of z values. A build's first layer is
z' = Lambda z (kernel K Lambda^T). Positions are weighted by rho (uniform).

Case A (no symmetry): N = 6 members, depth m = 3, n_y = 5 logits, 5 positions, random edges and head. Three gates, each
phi(t) = relu(alpha t + beta), reading one generic column each: S_1 reads z, S_2 reads v_1, S_3 reads v_2. Forced space
R = span{S_1, S_2, w_1^T S_3}, so r = 3.
Case B (H-eq, H-symN exact): N = 8, positions on a ring of 16 with gamma p = p + 2, z(gamma^q x, p) = Pi^q z(x, p - 2q)
(a symmetrised law built from 400 base inputs), circulant edges, readout with 2 orbits of 8 logits
(H[(s, j), i] = h_s[i - j]), every gate reads the orbit sum (S_l = 1).

Predictions (written before the run):
  A1  (T8 under self-gates) y = A uhat + c on every input, about 1e-13.
  A2  r = 3 (rank of the pulled-back reads).
  A3  (T11 parts 2, 3) at every k in {3, 4, 5, 6}: the T11 build, run as a self-contained network that computes its own
      gates, has gates equal to the network's (Lemma T11a, about 1e-13), D2 = 1/2 sum_{j > k - r} nu_j (about 1e-12
      relative), and centred output = C_yf C_f^-1 (f - E f) + U' U'^T e (about 1e-12).
  A4  (T11 part 1) 2000 random read-faithful builds per k (random invertible mixing of [R_0^T; random], random invertible
      edges, reads solved from the effective-kernel condition, best head and bias): their gates equal the network's,
      and none is below the formula (smallest ratio >= 1 - 1e-12).
  A5  (T11 part 4) running the stored width-6 build with only its first k units (and the width-k bias) gives the
      width-k value, about 1e-12.
  A6  (T11 part 5) at k = r the build is the forced reads alone and its value is 1/2 tr Cov(e).
  A7  (T11.1) widths (k + r, k + r, k) with the T9 read below the forced reads: gates equal the network's, and
      D2 = 1/2 sum_{j > k} mu_j at k = 1, 2, 3, 4 (about 1e-12 relative).
  A8  (T12) random self-contained builds with random reads (gates differ from the network's), and random non-linear
      feature maps of z with a width-k linear head: none below 1/2 sum_{j > k} mu_j. The PCA features U_k^T P_perp y reach
      it exactly. (No prediction for where the random-read builds sit relative to the T11 value: Conjecture D6.)
  B1  (T11.2 part 1) the gates are invariant: g_l(gamma x, gamma p) = g_l(x, p) (about 1e-13), and r = 1.
  B2  (T11.2 part 2) the non-zero nu_j equal the lambda_n of T10 for n != 0 (about 1e-12).
  B3  (T11.2 part 3) at every k: the T11 build value = 1/2 sum of lambda_n outside {0} and the top k - 1 other modes;
      at k = 1 the width-1 self-contained network (orbit-sum unit with its own gates) reaches 1/2 sum_{n != 0} lambda_n.
  B4  (T11.2 part 4) cost against T10 equals the stated difference, zero exactly when lambda_0 is among the top k.
"""
import numpy as np

rng = np.random.default_rng(11)


def relu(t):
    return np.maximum(t, 0.0)


def forward(z, Lam, ws, Ss, phis, Hm, c, rho):
    """A self-gated chain or a self-contained build: units z' = Lam z, gates from its own reads."""
    zp = np.einsum("ai,sip->sap", Lam, z)
    g = phis[0](np.einsum("ar,sap->srp", Ss[0], zp))
    gates = [g]
    v = g * zp
    for l, w in enumerate(ws):
        g = phis[l + 1](np.einsum("ar,sap->srp", Ss[l + 1], v))
        gates.append(g)
        v = g * np.einsum("ab,sbp->sap", w, v)
    y = np.einsum("ya,sap,p->sy", Hm, v, rho) + c
    return y, gates, v


def make_phi(alpha, beta):
    return lambda t: relu(alpha * t[:, 0:1, :] + beta)       # reads the single column (shape (S, 1, P))


def D2(y, yb, P):
    e = (yb - y) @ P
    return 0.5 * np.mean(np.sum(e ** 2, axis=1))


def cov(X, Y=None):
    Xc = X - X.mean(0)
    Yc = Xc if Y is None else Y - Y.mean(0)
    return Xc.T @ Yc / len(X)


def best_head_D2(feat, y, P):
    """Best linear head and bias on features feat (S, k): D2 of the least-squares fit of P_perp y."""
    F = np.concatenate([feat, np.ones((len(feat), 1))], axis=1)
    T = y @ P
    coef, *_ = np.linalg.lstsq(F, T, rcond=None)
    return 0.5 * np.mean(np.sum((F @ coef - T) ** 2, axis=1))


def analyse(z, ws, Ss, phis, Hm, c, rho, name):
    N = z.shape[1]
    ny = Hm.shape[0]
    P = np.eye(ny) - np.ones((ny, ny)) / ny
    I = np.eye(N)
    y, gates, _ = forward(z, I, ws, Ss, phis, Hm, c, rho)
    G = np.prod(np.stack([g[:, 0, :] for g in gates]), axis=0)               # (S, P)
    uhat = np.einsum("sip,sp,p->si", z, G, rho)
    A = Hm.copy()
    Wl = [I]
    for w in ws:
        Wl.append(w @ Wl[-1])
    A = Hm @ Wl[-1]
    out = {"A1_T8": np.abs(y - (uhat @ A.T + c)).max() / np.abs(y).max()}
    pulled = [Ss[0]] + [Wl[l].T @ Ss[l + 1] for l in range(len(ws))]
    Rall = np.concatenate(pulled, axis=1)
    U_, s_, _ = np.linalg.svd(Rall, full_matrices=False)
    r = int(np.sum(s_ > 1e-10 * s_[0]))
    R0 = U_[:, :r]
    Chat = cov(uhat)
    f = uhat @ R0
    Cf = R0.T @ Chat @ R0
    Cyf = P @ A @ Chat @ R0
    Cy = P @ A @ Chat @ A.T @ P
    mu = np.sort(np.linalg.eigvalsh(Cy))[::-1]
    Ce = Cy - Cyf @ np.linalg.solve(Cf, Cyf.T)
    nu, E = np.linalg.eigh(Ce)
    order = np.argsort(nu)[::-1]
    nu, E = nu[order], E[:, order]
    return dict(y=y, gates=gates, uhat=uhat, A=A, Wl=Wl, R0=R0, r=r, Chat=Chat, f=f, Cf=Cf, Cyf=Cyf, mu=mu, nu=nu, E=E,
                P=P, out=out, ny=ny, N=N)


def t11_build(an, k, ws, Ss, c):
    """Theorem T11 part 2: read [R0^T; U'^T (P A - Cyf Cf^-1 R0^T)], identity edges, reads M, head, bias."""
    R0, r, A, P, Cyf, Cf, E = an["R0"], an["r"], an["A"], an["P"], an["Cyf"], an["Cf"], an["E"]
    Up = E[:, :k - r]
    B = Cyf @ np.linalg.inv(Cf)
    Lam = np.concatenate([R0.T, Up.T @ (P @ A - B @ R0.T)], axis=0)
    Hp = np.concatenate([B, Up], axis=1)
    cp = c + (A - Hp @ Lam) @ an["uhat"].mean(0)
    targets = [Ss[0]] + [an["Wl"][l].T @ Ss[l + 1] for l in range(len(ws))]
    Ms = []
    for T in targets:
        cc, *_ = np.linalg.lstsq(R0, T, rcond=None)
        Ms.append(np.concatenate([cc, np.zeros((k - r, T.shape[1]))], axis=0))
    return Lam, [np.eye(k)] * len(ws), Ms, Hp, cp


def gate_err(g1, g2):
    return max(np.abs(a - b).max() for a, b in zip(g1, g2)) / max(np.abs(a).max() for a in g1)


# ---------------------------------------------------------------- case A
N, m, ny, Pn, S = 6, 3, 5, 5, 4000
rhoA = np.full(Pn, 1.0 / Pn)
zA = rng.normal(size=(S, N, Pn)) + 0.3 * rng.normal(size=(S, 1, Pn))
wsA = [rng.normal(size=(N, N)) / np.sqrt(N) for _ in range(m - 1)]
SsA = [rng.normal(size=(N, 1)) for _ in range(m)]
phisA = [make_phi(1.0, 0.4), make_phi(0.8, 0.5), make_phi(1.2, 0.3)]
HA = rng.normal(size=(ny, N))
cA = rng.normal(size=ny)
an = analyse(zA, wsA, SsA, phisA, HA, cA, rhoA, "A")
res = {"A1_T8_rel": an["out"]["A1_T8"], "A2_r": an["r"]}
r = an["r"]
res["nu"] = an["nu"].tolist()
res["mu"] = an["mu"].tolist()
A3 = {}
stored = t11_build(an, N, wsA, SsA, cA) if N - r <= np.sum(an["nu"] > 1e-12 * an["nu"][0]) else None
for k in range(r, min(N, r + int(np.sum(an["nu"] > 1e-12 * an["nu"][0]))) + 1):
    Lam, wsp, Ms, Hp, cp = t11_build(an, k, wsA, SsA, cA)
    yb, gb, _ = forward(zA, Lam, wsp, Ms, phisA, Hp, cp, rhoA)
    formula = 0.5 * np.sum(an["nu"][k - r:])
    val = D2(an["y"], yb, an["P"])
    cen = (yb - yb.mean(0)) @ an["P"]
    e = (an["y"] - an["y"].mean(0)) @ an["P"] - (an["f"] - an["f"].mean(0)) @ np.linalg.solve(an["Cf"], an["Cyf"].T)
    Up = an["E"][:, :k - r]
    pred = (an["f"] - an["f"].mean(0)) @ np.linalg.solve(an["Cf"], an["Cyf"].T) + e @ Up @ Up.T
    A3[k] = {"gate_err": gate_err(an["gates"], gb), "D2": val, "formula": formula,
             "rel": abs(val - formula) / max(formula, 1e-300), "centred_err": np.abs(cen - pred).max() / np.abs(pred).max()}
    if k == r:
        A3[k]["A6_half_tr_Ce"] = 0.5 * np.trace(cov(e))
res["A3"] = A3

# A4 random read-faithful builds
A4 = {}
for k in range(r, N + 1):
    ratios, gerr = [], 0.0
    formula = 0.5 * np.sum(an["nu"][k - r:])
    for t in range(2000 if k < N else 500):
        T = rng.normal(size=(k, k))
        Lam = T @ np.concatenate([an["R0"].T, rng.normal(size=(k - r, N))], axis=0)
        wsp = [rng.normal(size=(k, k)) for _ in range(m - 1)]
        Wp = [np.eye(k)]
        for w in wsp:
            Wp.append(w @ Wp[-1])
        targets = [SsA[0]] + [an["Wl"][l].T @ SsA[l + 1] for l in range(m - 1)]
        Ms = []
        Wfor = [Wp[0]] + Wp[:m - 1]                                 # gate 1 reads z', gate l + 1 reads v'_l
        for l, Tg in enumerate(targets):
            Mat = (Wfor[l] @ Lam).T                                   # (N, k): effective read kernels, z-level
            sol, *_ = np.linalg.lstsq(Mat, Tg, rcond=None)
            Ms.append(sol)
        yb, gb, vlast = forward(zA, Lam, wsp, Ms, phisA, np.eye(k), np.zeros(k), rhoA)
        gerr = max(gerr, gate_err(an["gates"], gb))
        ratios.append(best_head_D2(yb, an["y"], an["P"]) / formula)
    A4[k] = {"min_ratio": float(np.min(ratios)), "max_gate_err": gerr}
res["A4"] = A4

# A5 prefix run of the stored width-N build
A5 = {}
LamN, wsN, MsN, HN, cN = t11_build(an, N, wsA, SsA, cA)
for k in range(r, N + 1):
    Lam_k, _, Ms_k, H_k, c_k = t11_build(an, k, wsA, SsA, cA)
    yb, _, _ = forward(zA, LamN[:k], [np.eye(k)] * (m - 1), [M[:k] for M in MsN], phisA, HN[:, :k], c_k, rhoA)
    A5[k] = {"prefix_rel": abs(D2(an["y"], yb, an["P"]) - 0.5 * np.sum(an["nu"][k - r:])) / (0.5 * np.sum(an["nu"][k - r:])),
             "same_arrays": bool(np.allclose(Lam_k, LamN[:k], atol=1e-12) and np.allclose(H_k, HN[:, :k], atol=1e-12))}
res["A5"] = A5

# A7 Corollary T11.1
A7 = {}
mu = an["mu"]
evals, evecs = np.linalg.eigh(an["P"] @ an["A"] @ an["Chat"] @ an["A"].T @ an["P"])
o = np.argsort(evals)[::-1]
Ufull = evecs[:, o]
for k in range(1, ny):
    Uk = Ufull[:, :k]
    Lam9 = Uk.T @ an["P"] @ an["A"]
    Lam = np.concatenate([an["R0"].T, Lam9], axis=0)
    kk = k + r
    wsp = [np.eye(kk)] * (m - 2) + [np.concatenate([np.zeros((k, r)), np.eye(k)], axis=1)]
    targets = [SsA[0]] + [an["Wl"][l].T @ SsA[l + 1] for l in range(m - 1)]
    Ms = []
    for Tg in targets:
        cc, *_ = np.linalg.lstsq(an["R0"], Tg, rcond=None)
        Ms.append(np.concatenate([cc, np.zeros((k, Tg.shape[1]))], axis=0))
    cp = cA + (an["A"] - Uk @ Lam9) @ an["uhat"].mean(0)
    yb, gb, _ = forward(zA, Lam, wsp, Ms, phisA, Uk, cp, rhoA)
    formula = 0.5 * np.sum(mu[k:])
    A7[k] = {"gate_err": gate_err(an["gates"], gb), "rel": abs(D2(an["y"], yb, an["P"]) - formula) / formula,
             "T11_uniform_value": 0.5 * np.sum(an["nu"][max(k - r, 0):]) if k >= r else None, "T9_value": formula}
res["A7"] = A7

# A8 T12
A8 = {}
zflat = zA.reshape(S, -1)
for k in range(1, ny):
    bound = 0.5 * np.sum(mu[k:])
    rr, rs = [], []
    for t in range(300):
        Lam = rng.normal(size=(k, N))
        wsp = [rng.normal(size=(k, k)) for _ in range(m - 1)]
        Ms = [rng.normal(size=(k, 1)) for _ in range(m)]
        yb, _, _ = forward(zA, Lam, wsp, Ms, phisA, np.eye(k), np.zeros(k), rhoA)
        rs.append(best_head_D2(yb, an["y"], an["P"]) / bound)
        h = relu(zflat @ rng.normal(size=(zflat.shape[1], 64)) / 5 + rng.normal(size=64))
        feat = relu(h @ rng.normal(size=(64, k)))
        rr.append(best_head_D2(feat, an["y"], an["P"]) / bound)
    pca_feat = an["y"] @ an["P"] @ Ufull[:, :k]
    A8[k] = {"selfgate_random_reads_min_ratio": float(np.min(rs)), "random_relu_min_ratio": float(np.min(rr)),
             "pca_ratio": best_head_D2(pca_feat, an["y"], an["P"]) / bound,
             "T11_over_bound": (0.5 * np.sum(an["nu"][k - r:]) / bound) if k >= r else None,
             "selfgate_random_reads_below_T11": (float(np.min(rs)) * bound < 0.5 * np.sum(an["nu"][k - r:])) if k >= r else None}
res["A8"] = A8

# ---------------------------------------------------------------- case B
N, m, Pn, n0, sh = 8, 3, 16, 400, 2
rhoB = np.full(Pn, 1.0 / Pn)
z0 = rng.normal(size=(n0, N, Pn)) + 0.5 * rng.normal(size=(n0, 1, Pn)) + 0.3
Pi = np.roll(np.eye(N), 1, axis=0)


def act(z, q):
    """z(gamma^q x, p) = Pi^q z(x, p - sh q)."""
    return np.roll(np.roll(z, q, axis=1), sh * q, axis=2)


zB = np.concatenate([act(z0, q) for q in range(N)], axis=0)


def circulant(cv):
    return np.stack([np.roll(cv, i) for i in range(len(cv))], axis=0)


wsB = [circulant(rng.normal(size=N)) / np.sqrt(N) for _ in range(m - 1)]
SsB = [np.ones((N, 1)) / N for _ in range(m)]
phisB = [make_phi(1.0, 0.4), make_phi(0.8, 0.5), make_phi(1.2, 0.3)]
n_orb = 2
hs = rng.normal(size=(n_orb, N))
HB = np.array([[hs[s][(i - j) % N] for i in range(N)] for s in range(n_orb) for j in range(N)])
cB = np.tile(rng.normal(size=n_orb)[:, None], (1, N)).ravel()
bn = analyse(zB, wsB, SsB, phisB, HB, cB, rhoB, "B")
resB = {"A1_T8_rel": bn["out"]["A1_T8"], "B1_r": bn["r"]}
# gate invariance: gates of act(z, 1) at gamma p vs gates of z at p
_, g_shift, _ = forward(act(z0, 1), np.eye(N), wsB, SsB, phisB, HB, cB, rhoB)
_, g_base, _ = forward(z0, np.eye(N), wsB, SsB, phisB, HB, cB, rhoB)
resB["B1_gate_invariance"] = max(np.abs(gs - np.roll(gb_, sh, axis=2)).max() for gs, gb_ in zip(g_shift, g_base))
# T10 lambda_n
ev, V = np.linalg.eigh(bn["Chat"])
Ch_half = V @ np.diag(np.sqrt(np.maximum(ev, 0))) @ V.T
Phi = bn["A"].T @ bn["P"] @ bn["A"]
M = Ch_half @ Phi @ Ch_half
F = np.fft.fft(np.eye(N)) / np.sqrt(N)
lam = np.real(np.diag(F.conj().T @ M @ F))            # lambda_n on mode n (circulant M)
resB["B_circulant_err"] = np.abs(F.conj().T @ M @ F - np.diag(np.diag(F.conj().T @ M @ F))).max() / np.abs(M).max()
lam_other = np.sort(lam[1:])[::-1]
nu_nz = bn["nu"][: N - 1]
resB["B2_nu_vs_lambda"] = float(np.abs(np.sort(nu_nz)[::-1] - lam_other).max() / lam_other.max())
resB["lambda"] = lam.tolist()
B3 = {}
for k in range(1, N + 1):
    Lam, wsp, Ms, Hp, cp = t11_build(bn, k, wsB, SsB, cB)
    yb, gb, _ = forward(zB, Lam, wsp, Ms, phisB, Hp, cp, rhoB)
    val = D2(bn["y"], yb, bn["P"])
    pred = 0.5 * np.sum(lam_other[k - 1:])
    t10 = 0.5 * np.sum(np.sort(lam)[::-1][k:])
    cost_formula = np.sum(np.sort(lam)[::-1][:k]) - (lam[0] + np.sum(lam_other[:k - 1]))
    B3[k] = {"gate_err": gate_err(bn["gates"], gb), "D2": val, "formula": pred,
             "rel": abs(val - pred) / max(pred, 1e-12 * lam.max()), "T10_value": t10,
             "B4_cost_err": abs((val - t10) - 0.5 * cost_formula) / lam.max(),
             "lambda0_in_topk": bool(lam[0] >= np.sort(lam)[::-1][k - 1] - 1e-12 * lam.max())}
resB["B3"] = B3
res["B"] = resB


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


import json
print(json.dumps(clean(res), indent=1))
json.dump(clean(res), open(__file__.replace(".py", ".json"), "w"), indent=1)
