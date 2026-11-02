# serves: T9
# serves: T10
# serves: T6
"""Exact instance of a gated chain (COARSE-THEORY.md B.4). Checks Lemma T8, Theorems T6, T9, T10, Corollaries
T10.1 and T10.2 and Fact F11 to machine precision.

The instance satisfies H-gate, H-eq and H-symN exactly. Z_N (N = 8) acts on a polar grid by a shift of 2 angle bins
(16 bins). Inputs live on 3 rings x 16 bins, positions on 2 rings x 16 bins. The first family has kernels
K(i, p) = gamma^i K(0, gamma^-i p) (one free kernel per position for member 0). Three gates per position come from
invariant kernels (orbit averages) through a ReLU: the gate path. Edges w_1, w_2 are random circulant matrices. The
readout has 2 orbits of 8 logits, H[(s, j), i] = h_s[i - j], so H Pi = R_y H. The empirical law is symmetrised over
the group, so H-symN holds exactly for it.

Predictions (written before the run):
  P1  Chat and Phi are circulant and E uhat is constant (relative error about 1e-14).
  P2  the eigenvalues of Chat^1/2 Phi Chat^1/2 are chat(n) phihat(n) on the Fourier modes (about 1e-12).
  P3  the non-zero eigenvalues mu_j of Cov(P_perp y), from a forward pass, equal the lambda_n (about 1e-12).
  P4  at every width k with a strict gap: the T9 build (read U_k' P_perp A, identity edges) and the T10 build
      (Q_S at every layer) both reach D2 = 1/2 sum_{j>k} mu_j in a real forward pass; their centred outputs agree
      on every input; the T10 units equal Q_S' v_l at every layer (all about 1e-12).
  P5  no build of the same width does better: 2000 random builds (with their best bias) are all above the formula,
      and a random perturbation of size eps raises D2 by a positive amount that scales as eps^2.
  P6  nesting: U_k U_k' U_k2 U_k2' = U_k U_k' and merges compose (about 1e-14).
  P7  Fact F11: KL <= (1/4) |P_perp delta|^2 on every input for every build tried (no exception).
  P8  Corollary T10.1: with an invariant readout (R_y = I) the width-1 orbit-sum build has D2 = 0 (relative 1e-13).
  P9  control that breaks H-symN (mixture of the symmetric law and an unsymmetrised sample, weight delta): T9 stays
      exact (P4 for the T9 build, about 1e-12), but the harmonic T10 build has an excess over the T9 optimum that
      grows as delta^2 (log-log slope 2 within 0.1) for small delta.
"""
import numpy as np

rng = np.random.default_rng(1)
N, Bn, shift = 8, 16, 2
Ji, Jp = 3, 2
D = Ji * Bn + 1                                  # input pixels + constant entry
Pn = Jp * Bn
rho = np.full(Pn, 1.0 / Pn)


def g_in(xt, k=1):
    """gamma^k on an augmented input (..., D): roll the angle bins of the pixel part."""
    x = xt[..., :-1].reshape(xt.shape[:-1] + (Ji, Bn))
    x = np.roll(x, k * shift, axis=-1)
    return np.concatenate([x.reshape(xt.shape[:-1] + (Ji * Bn,)), xt[..., -1:]], axis=-1)


def g_pos(p, k=1):
    j, b = divmod(p, Bn)
    return j * Bn + (b + k * shift) % Bn


# first family: K(i, p) = gamma^i K(0, gamma^-i p)
K0 = rng.normal(size=(Pn, D)) * 0.3
K0[:, -1] = rng.normal(size=Pn) * 0.2
Kmem = np.zeros((N, Pn, D))
for i in range(N):
    for p in range(Pn):
        Kmem[i, p] = g_in(K0[g_pos(p, -i)], i)


def invariant_kernel():
    k0 = rng.normal(size=(Pn, D)) * 0.3
    k0[:, -1] = 0.3
    out = np.zeros_like(k0)
    for j in range(N):
        for p in range(Pn):
            out[p] += g_in(k0[g_pos(p, -j)], j) / N
    return out


Kg = [invariant_kernel() for _ in range(3)]


def circulant(c):
    return np.stack([np.roll(c, i) for i in range(len(c))], axis=0)


w1 = circulant(rng.normal(size=N))
w2 = circulant(rng.normal(size=N))
n_orb = 2
hs = rng.normal(size=(n_orb, N))
H = np.array([[hs[s][(i - j) % N] for i in range(N)] for s in range(n_orb) for j in range(N)])   # (16, N)
cvec = np.tile(rng.normal(size=n_orb)[:, None], (1, N)).ravel()
ny = H.shape[0]
Pperp = np.eye(ny) - np.ones((ny, ny)) / ny
Pi = np.roll(np.eye(N), 1, axis=0)               # (Pi v)_{i+1} = v_i


def sample(S0, sym=True):
    x = np.abs(rng.normal(size=(S0, Ji * Bn))) ** 1.5 + 0.2 * rng.normal(size=(S0, Ji * Bn))
    xt = np.concatenate([x, np.ones((S0, 1))], axis=1)
    if sym:
        xt = np.concatenate([g_in(xt, k) for k in range(N)], axis=0)
    return xt


def gates(xt):
    return [np.maximum(np.einsum("sd,pd->sp", xt, k), 0.0) for k in Kg]           # 3 x (S, P)


def teacher(xt, Hm=H, c=cvec):
    g = gates(xt)
    z = np.einsum("sd,ipd->sip", xt, Kmem)
    v1 = g[0][:, None, :] * z
    v2 = g[1][:, None, :] * np.einsum("ij,sjp->sip", w1, v1)
    v3 = g[2][:, None, :] * np.einsum("ij,sjp->sip", w2, v2)
    y = np.einsum("yi,sip,p->sy", Hm, v3, rho) + c
    G = g[0] * g[1] * g[2]
    uhat = np.einsum("sip,sp,p->si", z, G, rho)
    return y, uhat, (v1, v2, v3)


def build(xt, Kp, w1p, w2p, Hp, cp):
    g = gates(xt)
    c1 = g[0][:, None, :] * np.einsum("sd,apd->sap", xt, Kp)
    c2 = g[1][:, None, :] * np.einsum("ab,sbp->sap", w1p, c1)
    c3 = g[2][:, None, :] * np.einsum("ab,sbp->sap", w2p, c2)
    return np.einsum("ya,sap,p->sy", Hp, c3, rho) + cp, (c1, c2, c3)


def D2(y, yb, wts):
    e = (yb - y) @ Pperp
    return 0.5 * np.sum(wts * np.sum(e ** 2, axis=1))


def wcov(X, wts):
    m = wts @ X
    Xc = X - m
    return (Xc * wts[:, None]).T @ Xc, m


def best_bias(y, yb_nobias, wts):
    return wts @ (y - yb_nobias)


def real_fourier_basis(N):
    """Columns: mode 0, cos/sin pairs n = 1..N/2-1, mode N/2. Returns basis and list of mode labels."""
    t = 2 * np.pi * np.arange(N) / N
    cols, labels = [np.ones(N) / np.sqrt(N)], [0]
    for n in range(1, N // 2):
        cols += [np.cos(n * t) * np.sqrt(2 / N), np.sin(n * t) * np.sqrt(2 / N)]
        labels += [n, n]
    cols.append(np.cos(N // 2 * t) / np.sqrt(N)); labels.append(N // 2)
    return np.stack(cols, axis=1), np.array(labels)


def rel(a, b):
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)


def circ_err(M):
    return rel(Pi @ M @ Pi.T, M)


def kl_rows(y, yb):
    a = y - y.max(1, keepdims=True); b = yb - yb.max(1, keepdims=True)
    lq = a - np.log(np.exp(a).sum(1, keepdims=True)); lp = b - np.log(np.exp(b).sum(1, keepdims=True))
    return np.sum(np.exp(lq) * (lq - lp), axis=1)


def analyse(xt, wts, Hm=H, c=cvec, label="", ks=None, report=True):
    y, uhat, vs = teacher(xt, Hm, c)
    A = Hm @ w2 @ w1
    Chat, Eu = wcov(uhat, wts)
    Phi = A.T @ Pperp @ A
    covy, Ey = wcov(y @ Pperp, wts)
    mu, E = np.linalg.eigh(covy); order = np.argsort(mu)[::-1]; mu, E = mu[order], E[:, order]
    res = {"y": y, "uhat": uhat, "vs": vs, "A": A, "Chat": Chat, "Eu": Eu, "Phi": Phi, "mu": mu, "E": E}
    return res


out = {}
X = sample(300)
S = len(X)
wts = np.full(S, 1.0 / S)
R = analyse(X, wts)
y, uhat, A, Chat, Phi, mu, E = R["y"], R["uhat"], R["A"], R["Chat"], R["Phi"], R["mu"], R["E"]

# P1
out["P1 Chat circulant"] = circ_err(Chat)
out["P1 Phi circulant"] = circ_err(Phi)
out["P1 E uhat constant"] = np.ptp(R["Eu"]) / np.abs(R["Eu"]).max()

# P2: lambda_n from the Fourier modes vs eigenvalues of Chat^1/2 Phi Chat^1/2
ev, V = np.linalg.eigh(Chat)
Ch = V @ np.diag(np.sqrt(ev)) @ V.T
Ainner = Ch @ Phi @ Ch
F, labels = real_fourier_basis(N)
lam_modes = np.diag(F.T @ Ainner @ F)
chat_n = np.diag(F.T @ Chat @ F); phihat_n = np.diag(F.T @ Phi @ F)
out["P2 off-diagonal in Fourier basis"] = np.abs(F.T @ Ainner @ F - np.diag(lam_modes)).max() / np.abs(lam_modes).max()
out["P2 lambda_n = chat(n) phihat(n)"] = rel(lam_modes, chat_n * phihat_n)
eig_inner = np.sort(np.linalg.eigvalsh(Ainner))[::-1]
out["P2 eigenvalues = mode values"] = rel(eig_inner, np.sort(lam_modes)[::-1])

# P3
out["P3 mu (logit PCA) = lambda"] = rel(mu[:N], eig_inner)

# widths with a strict gap: block boundaries along the order of lambda_n
modes = sorted(set(labels.tolist()), key=lambda n: -lam_modes[labels == n][0])
widths, Ssets, acc = [], [], []
for n in modes:
    acc = acc + [n]
    widths.append(int(sum((labels == m).sum() for m in acc)))
    Ssets.append(list(acc))
gaps = [mu[k - 1] - mu[k] for k in widths if k < N]
out["mode order (largest lambda first)"] = modes
out["widths with a strict gap"] = widths
out["smallest relative gap at those widths"] = min(gaps) / mu[0]

# P4
P4 = {}
for k, Sset in zip(widths, Ssets):
    if k >= N:
        continue
    Uk = E[:, :k]
    Lam = Uk.T @ Pperp @ A
    Kp = np.einsum("ai,ipd->apd", Lam, Kmem)
    yb0, _ = build(X, Kp, np.eye(k), np.eye(k), Uk, np.zeros(ny))
    cp = best_bias(y, yb0, wts)
    yb9 = yb0 + cp
    target = 0.5 * mu[k:].sum()
    d9 = D2(y, yb9, wts)
    Q = F[:, np.isin(labels, Sset)]
    Kq = np.einsum("ia,ipd->apd", Q, Kmem)
    yb10, cs = build(X, Kq, Q.T @ w1 @ Q, Q.T @ w2 @ Q, H @ Q, np.zeros(ny))
    yb10 = yb10 + cvec + A @ (np.eye(N) - Q @ Q.T) @ R["Eu"]
    d10 = D2(y, yb10, wts)
    unit_err = max(rel(cs[l], np.einsum("ia,sip->sap", Q, R["vs"][l])) for l in range(3))
    cen9 = (yb9 - wts @ yb9) @ Pperp; cen10 = (yb10 - wts @ yb10) @ Pperp
    cen_pred = (uhat - R["Eu"]) @ (Pperp @ A @ Q @ Q.T).T
    P4[k] = {"D2 T9 vs formula": abs(d9 - target) / target, "D2 T10 vs formula": abs(d10 - target) / target,
             "centred outputs T9 = T10": rel(cen9, cen10), "centred output = P_perp A P_S uhat": rel(cen10, cen_pred),
             "T10 units = Q_S' v_l": unit_err, "formula D2 / D2 at k=0": target / (0.5 * mu.sum())}
    if k == widths[1]:
        keep = dict(k=k, Kq=Kq, Q=Q, d=d10)
out["P4"] = P4

# P5: random builds and perturbations at the second width
k, Q, Kq, dstar = keep["k"], keep["Q"], keep["Kq"], keep["d"]
Xs = X[::4]; ws = np.full(len(Xs), 1.0 / len(Xs)); ys = teacher(Xs)[0]


def d2_of(Kp, a, b, Hp):
    yb, _ = build(Xs, Kp, a, b, Hp, np.zeros(ny))
    return D2(ys, yb + best_bias(ys, yb, ws), ws)


base = d2_of(Kq, Q.T @ w1 @ Q, Q.T @ w2 @ Q, H @ Q)
rand = []
for t in range(2000):
    sc = rng.uniform(0.1, 2.0)
    rand.append(d2_of(rng.normal(size=Kq.shape) * 0.3 * sc, rng.normal(size=(k, k)), rng.normal(size=(k, k)),
                      rng.normal(size=(ny, k))))
out["P5 random builds: min D2 / formula D2 (must be >= 1)"] = min(rand) / base
xi = [rng.normal(size=Kq.shape) * 0.3, rng.normal(size=(k, k)), rng.normal(size=(k, k)), rng.normal(size=(ny, k))]
rises = []
for eps in [1e-2, 3e-3, 1e-3]:
    rises.append(d2_of(Kq + eps * xi[0], Q.T @ w1 @ Q + eps * xi[1], Q.T @ w2 @ Q + eps * xi[2],
                       H @ Q + eps * xi[3]) - base)
out["P5 perturbation rises (eps = 1e-2, 3e-3, 1e-3)"] = rises
out["P5 log-log slope"] = np.polyfit(np.log([1e-2, 3e-3, 1e-3]), np.log(np.maximum(rises, 1e-300)), 1)[0]

# P6 nesting and merges
P6 = 0.0
ws_list = [kk for kk in widths if kk < N]
for a_, b_ in zip(ws_list[:-1], ws_list[1:]):
    U1, U2 = E[:, :a_], E[:, :b_]
    P6 = max(P6, np.abs(U1 @ U1.T @ U2 @ U2.T - U1 @ U1.T).max())
if len(ws_list) >= 3:
    U1, U2, U3 = E[:, :ws_list[0]], E[:, :ws_list[1]], E[:, :ws_list[2]]
    P6 = max(P6, np.abs((U1.T @ U2) @ (U2.T @ U3) - U1.T @ U3).max())
out["P6 nesting and merge errors"] = P6

# P7 Fact F11 on every input, for every build used above and the random ones
viol = 0
worst = 0.0
for t in range(200):
    yb, _ = build(Xs, rng.normal(size=Kq.shape) * 0.3, rng.normal(size=(k, k)), rng.normal(size=(k, k)),
                  rng.normal(size=(ny, k)), rng.normal(size=ny))
    kl = kl_rows(ys, yb); bound = 0.25 * np.sum(((yb - ys) @ Pperp) ** 2, axis=1)
    viol += int(np.sum(kl > bound + 1e-12)); worst = max(worst, np.max(kl / np.maximum(bound, 1e-300)))
out["P7 F11 violations (inputs x builds)"] = viol
out["P7 largest KL / bound"] = worst

# P8 invariant readout
Hinv = np.tile(rng.normal(size=(ny, 1)), (1, N))
Ri = analyse(X, wts, Hm=Hinv)
Q0 = F[:, labels == 0]
yb, _ = build(X, np.einsum("ia,ipd->apd", Q0, Kmem), Q0.T @ w1 @ Q0, Q0.T @ w2 @ Q0, Hinv @ Q0, np.zeros(ny))
yb = yb + best_bias(Ri["y"], yb, wts)
out["P8 width-1 D2 / D2 at k=0 (invariant readout)"] = D2(Ri["y"], yb, wts) / (0.5 * Ri["mu"].sum())
# P8b: T10.1 as restated (only H-eq parts 3-4): untied random member kernels, unsymmetrised law. The width-1 build
# (kernel sum_i K_i, edges = column sums s_l, head alpha') must give y exactly.
Kfree = rng.normal(size=(N, Pn, D)) * 0.3
Xu = sample(300, sym=False); gu = gates(Xu)
zu = np.einsum("sd,ipd->sip", Xu, Kfree)
v1 = gu[0][:, None, :] * zu
v3 = gu[2][:, None, :] * np.einsum("ij,sjp->sip", w2, gu[1][:, None, :] * np.einsum("ij,sjp->sip", w1, v1))
yu = np.einsum("yi,sip,p->sy", Hinv, v3, rho) + cvec
s1, s2 = w1.sum(0)[0], w2.sum(0)[0]
yb1, _ = build(Xu, Kfree.sum(0)[None], np.array([[s1]]), np.array([[s2]]), Hinv[:, :1], cvec)
out["P8b width-1 exact, untied kernels, unsymmetrised law (max rel error of logits)"] = rel(yb1, yu)

# P9 symmetry-breaking control
Xa = sample(200, sym=False)
Xmix = np.concatenate([X, Xa], axis=0)
k9, S9 = widths[1], Ssets[1]
Q9 = F[:, np.isin(labels, S9)]
exc, exact9 = [], []
deltas = [1e-3, 3e-3, 1e-2, 3e-2]
for dl in deltas:
    wm = np.concatenate([np.full(len(X), (1 - dl) / len(X)), np.full(len(Xa), dl / len(Xa))])
    Rm = analyse(Xmix, wm)
    Uk = Rm["E"][:, :k9]
    Lam = Uk.T @ Pperp @ Rm["A"]
    yb0, _ = build(Xmix, np.einsum("ai,ipd->apd", Lam, Kmem), np.eye(k9), np.eye(k9), Uk, np.zeros(ny))
    dopt = D2(Rm["y"], yb0 + best_bias(Rm["y"], yb0, wm), wm)
    exact9.append(abs(dopt - 0.5 * Rm["mu"][k9:].sum()) / dopt)
    yh, _ = build(Xmix, np.einsum("ia,ipd->apd", Q9, Kmem), Q9.T @ w1 @ Q9, Q9.T @ w2 @ Q9, H @ Q9, np.zeros(ny))
    dh = D2(Rm["y"], yh + best_bias(Rm["y"], yh, wm), wm)
    exc.append(dh - dopt)
out["P9 T9 exact under broken symmetry (max rel error)"] = max(exact9)
out["P9 harmonic excess (delta = 1e-3 .. 3e-2)"] = exc
out["P9 log-log slope"] = np.polyfit(np.log(deltas), np.log(np.maximum(exc, 1e-300)), 1)[0]

for key, val in out.items():
    if isinstance(val, dict):
        print(key)
        for kk, vv in val.items():
            print("   k =", kk, {a: float(f"{b:.3g}") for a, b in vv.items()})
    else:
        print(key, ":", val if not isinstance(val, float) else f"{val:.3g}")
