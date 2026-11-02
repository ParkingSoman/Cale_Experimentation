# serves: TV15
"""Exact checks of Part B.V.4 (COARSE-THEORY.md): Definition DV4, Lemmas TV15.1, TV15.2, TV13.1, Theorem TV15 and
Corollary TV15.3. Dense random pure-stream networks (d = 10, three blocks of 12 neurons, C = 4 logits), both forms of
Definition DV1, four activations (tanh, GELU, ReLU, leaky ReLU).
  A  TV15.1: the remainder modulus against its three formulas (smooth: (Lip'/2) t^2; differentiable point: o(t);
     kinks: 0 inside the margin, (1 - alpha) max(t - |a|, 0) for leaky ReLU)
  B  TV15.2: the second derivative of the norm by sympy against the formula; its supremum 2 / 3^(1/2); part 3's two
     bounds at random z, delta, eps (also with beta > |delta|)
  C  TV13.1: the exact identity with the norm's secant matrix (by quadrature), both forms, every activation
  D  TV15: the pointwise inequality and the D_P bound on random re-fit cuts
  E  TV15.3: the scaled formulas (rem(t) <= t^2 Rbar for smooth sigma; activation part 0 below t_1 for ReLU), and a
     family N_t whose D-content shrinks: for ReLU without norms, below t_1 the true change equals Lambda_N
Run: uv run --with sympy --with numpy python model/tv_norm.py [instances]"""
import sys
import numpy as np
import sympy as sp

rng = np.random.default_rng(1529)
N_INST = int(sys.argv[1]) if len(sys.argv) > 1 else 40
d, C, NX, L, q, DX = 10, 4, 64, 3, 12, 6
Pperp = np.eye(C) - np.ones((C, C)) / C
fails, notes = [], []
KG = np.sqrt(2 / np.pi)
ALPHA = 0.1


COUNT = {}


def check(name, ok, info=""):
    COUNT[name.split()[0]] = COUNT.get(name.split()[0], 0) + 1
    if not ok:
        fails.append(f"{name} {info}")


# ---------- activations: value, chosen slope, Lipschitz constant, remainder modulus omega(a, t) ----------
def gelu(x):
    return 0.5 * x * (1 + np.tanh(KG * (x + 0.044715 * x ** 3)))


def gelu_d(x):
    u = KG * (x + 0.044715 * x ** 3)
    th = np.tanh(u)
    return 0.5 * (1 + th) + 0.5 * x * (1 - th ** 2) * KG * (1 + 3 * 0.044715 * x ** 2)


GRID = np.linspace(-30, 30, 600001)
LIP_GELU = float(np.abs(gelu_d(GRID)).max()) * (1 + 1e-6)
LIPP_GELU = float(np.abs(np.diff(gelu_d(GRID)) / np.diff(GRID)).max()) * (1 + 1e-3)
LIPP_TANH = 4 / (3 * np.sqrt(3))

ACT = {
    "tanh": (np.tanh, lambda a: 1 - np.tanh(a) ** 2, 1.0),
    "gelu": (gelu, gelu_d, LIP_GELU),
    "relu": (lambda a: np.maximum(a, 0), lambda a: (a > 0).astype(float), 1.0),
    "leaky": (lambda a: np.maximum(a, ALPHA * a), lambda a: np.where(a > 0, 1.0, ALPHA), 1.0),
}
UG = np.linspace(-1, 1, 2001)


def omega(kind, a, t):
    """Definition DV4 at N's pre-activations a (any shape), window t (same shape), chosen slope ACT[kind][1](a)."""
    f, df, _ = ACT[kind]
    s = df(a)
    if kind in ("relu", "leaky"):
        al = 0.0 if kind == "relu" else ALPHA
        # Lemma TV15.1 part 3 (a = 0 does not occur on these nets with probability 1; it is checked below)
        return (1 - al) * np.maximum(t - np.abs(a), 0)
    if kind == "tanh":   # sup at u = +-t or at the critical point u = -2a (tanh'(a + u) = tanh'(a) iff a + u = +-a)
        r = lambda u: np.abs(f(a + u) - f(a) - s * u)
        cand = np.maximum(r(t), r(-t))
        return np.where(2 * np.abs(a) <= t, np.maximum(cand, r(-2 * a)), cand)
    # GELU: dense grid on [-t, t] with the ends; a tiny slack covers the grid
    A, T = np.asarray(a)[..., None], np.asarray(t)[..., None]
    U = UG * T
    vals = np.abs(f(A + U) - f(A) - df(A) * U).max(-1)
    return vals * (1 + 1e-6) + 1e-15


# ---------- networks ----------
def nrm(h, eps, on):
    return h / np.sqrt((h * h).sum(-1, keepdims=True) + eps) if on else h


def logsm(y):
    m = y.max(-1, keepdims=True)
    return y - m - np.log(np.exp(y - m).sum(-1, keepdims=True))


def kl(p, yN, yB):
    a, b = logsm(yN), logsm(yB)
    return float((p * (np.exp(a) * (a - b)).sum(-1)).sum())


def make_net(norm, kind):
    net = dict(act=kind, eps=[0.5] * (L + 1), norm=norm)
    X = rng.normal(size=(NX, DX))
    net["e"] = np.tanh(X @ rng.normal(size=(DX, d))) * 2 + X[:, :1] * 0.5
    net["R"] = [rng.normal(size=(q, d)) / np.sqrt(d) * (3 if norm else 1) for _ in range(L)]
    net["b"] = [rng.normal(size=q) * .3 for _ in range(L)]
    net["W"] = [rng.normal(size=(d, q)) / np.sqrt(q) * .7 for _ in range(L)]
    net["c"] = [rng.normal(size=d) * .1 for _ in range(L)]
    net["U"] = rng.normal(size=(C, d)) * (2 if norm else .5)
    net["cy"] = rng.normal(size=C)
    return net


def forward_N(net):
    f = ACT[net["act"]][0]
    h = net["e"]
    hs, ns, As = [h], [], []
    for l in range(L):
        z = nrm(h, net["eps"][l], net["norm"])
        a = z @ net["R"][l].T + net["b"][l]
        ns.append(z)
        As.append(a)
        h = h + f(a) @ net["W"][l].T + net["c"][l]
        hs.append(h)
    zL = nrm(h, net["eps"][L], net["norm"])
    return hs, ns, As, zL @ net["U"].T + net["cy"]


def make_cut(net, Pi, noise):
    """A re-fit area cut: reads and logits on D perp (projection reads plus noise), N's biases plus noise,
    norm constants eps_l + (a random c_l >= 0)."""
    cut = dict(R=[(net["R"][l] + noise * rng.normal(size=(q, d)) / np.sqrt(d)) @ Pi for l in range(L)],
               b=[net["b"][l] + noise * rng.normal(size=q) * .3 for l in range(L)],
               U=(net["U"] + noise * rng.normal(size=(C, d))) @ Pi,
               cy=net["cy"] + noise * rng.normal(size=C),
               eps=[net["eps"][l] + rng.uniform(0, 1) for l in range(L + 1)])
    return cut


def forward_cut(net, cut, Pi):
    f = ACT[net["act"]][0]
    h = net["e"] @ Pi.T
    hs = [h]
    for l in range(L):
        z = nrm(h, cut["eps"][l], net["norm"])
        h = h + (f(z @ cut["R"][l].T + cut["b"][l]) @ net["W"][l].T + net["c"][l]) @ Pi.T
        hs.append(h)
    return hs, nrm(h, cut["eps"][L], net["norm"]) @ cut["U"].T + cut["cy"]


def Dn(z, eps):
    """Lemma TV15.2 part 1, per row of z: (I - n n^T) / nu."""
    nu = np.sqrt((z * z).sum(-1) + eps)
    n = z / nu[:, None]
    return (np.eye(z.shape[1])[None] - np.einsum("xi,xj->xij", n, n)) / nu[:, None, None]


def leaks_omega(net, cut, Pi, hs, ns, yN):
    norm = net["norm"]
    lk = []
    for l in range(L):
        zc = nrm(hs[l] @ Pi.T, cut["eps"][l], norm)
        lk.append(zc @ cut["R"][l].T + cut["b"][l] - (ns[l] @ net["R"][l].T + net["b"][l]))
    om = (nrm(hs[L] @ Pi.T, cut["eps"][L], norm) @ cut["U"].T + cut["cy"] - yN) @ Pperp.T
    return lk, om


def tv15(net, cut, Pi, hs, As, lk, om, tscale=1.0, smooth_bar=False, tight=False):
    """Theorem TV15: Lambda_N and rem at each x, from N, P, D and the cut's reads (no run of the cut).
    tscale multiplies every leak and omega (Corollary TV15.3). smooth_bar: use (Lip'/2) t^2 in place of omega."""
    kind, norm = net["act"], net["norm"]
    df, Lip = ACT[kind][1], ACT[kind][2]
    lk = [tscale * x for x in lk]
    om = tscale * om
    I = np.eye(d)[None]
    H = [Dn(hs[l] @ Pi.T, cut["eps"][l]) if norm else np.repeat(I, NX, 0) for l in range(L + 1)]
    S = [df(As[l]) for l in range(L)]
    PW = [Pi @ net["W"][l] for l in range(L)]
    dN = np.zeros((NX, d))
    beta, gam = np.zeros(NX), np.zeros(NX)
    wsum = np.zeros(NX)
    for l in range(L):
        Rc = cut["R"][l]
        z = hs[l] @ Pi.T
        if norm:
            m = np.sqrt(np.maximum(np.linalg.norm(z, axis=1) - beta, 0) ** 2 + cut["eps"][l])
            kap, th = 1 / m, beta ** 2 / (np.sqrt(3) * m ** 2)
        else:
            kap, th = np.ones(NX), np.zeros(NX)
        tt = np.abs(np.linalg.norm(Rc, axis=1))[None, :] * (kap * beta)[:, None] + np.abs(lk[l])
        if smooth_bar:
            lipp = LIPP_TANH if kind == "tanh" else LIPP_GELU
            w = lipp / 2 * tt ** 2
        else:
            w = omega(kind, As[l], tt)
        wsum += np.linalg.norm(w, axis=1)
        J = np.einsum("dq,xq,qe,xef->xdf", PW[l], S[l], Rc, H[l])
        WSR = np.einsum("dq,xq,qe->xde", PW[l], S[l], Rc)
        nIJ = np.linalg.norm(I + J, ord=2, axis=(1, 2))
        nWSR = np.linalg.norm(WSR, ord=2, axis=(1, 2))
        nW, nR = np.linalg.norm(PW[l], 2), np.linalg.norm(Rc, 2)
        gam = nIJ * gam + nWSR * th + nW * np.linalg.norm(w, axis=1)
        dN = dN + (((np.einsum("xd,xde->xe", dN, H[l]) @ Rc.T + lk[l]) * S[l]) @ PW[l].T)
        beta = beta + Lip * nW * (nR * kap * beta + np.linalg.norm(lk[l], axis=1))
        if tight:   # Theorem TV15 part 2: |delta_l| <= |delta^N_l| + gamma_l
            beta = np.minimum(beta, np.linalg.norm(dN, axis=1) + gam)
    zL = hs[L] @ Pi.T
    PU = Pperp @ cut["U"]
    if norm:
        m = np.sqrt(np.maximum(np.linalg.norm(zL, axis=1) - beta, 0) ** 2 + cut["eps"][L])
        thL = beta ** 2 / (np.sqrt(3) * m ** 2)
    else:
        thL = np.zeros(NX)
    Lam = om + np.einsum("xd,xde->xe", dN, H[L]) @ PU.T
    nPUH = np.linalg.norm(np.einsum("cd,xde->xce", PU, H[L]), ord=2, axis=(1, 2))
    rem = nPUH * gam + np.linalg.norm(PU, 2) * thL
    return Lam, rem, wsum


def rand_D(k=3):
    Qd = np.linalg.qr(rng.normal(size=(d, k)))[0]
    PD = Qd @ Qd.T
    return PD, np.eye(d) - PD


def wnorm(p, f):
    return np.sqrt(float((p * f ** 2).sum()))


# ---------- A: Lemma TV15.1 ----------
for kind in ACT:
    f, df, Lip = ACT[kind]
    a = rng.normal(size=400) * 2
    t = rng.uniform(0, 3, size=400)
    om_ = omega(kind, a, t)
    brute = np.abs(f(a[:, None] + UG[None] * t[:, None]) - f(a)[:, None] - df(a)[:, None] * UG[None] * t[:, None]).max(1)
    check(f"A {kind} omega >= brute force", (om_ >= brute * (1 - 1e-9) - 1e-12).all())
    check(f"A {kind} omega <= 2 Lip t", (om_ <= 2 * Lip * t * (1 + 1e-6) + 1e-12).all())
    if kind in ("tanh", "gelu"):
        lipp = LIPP_TANH if kind == "tanh" else LIPP_GELU
        check(f"A {kind} smooth part 1", (om_ <= lipp / 2 * t ** 2 * (1 + 1e-6) + 1e-12).all())
        small = omega(kind, a, np.full(400, 1e-4)) / 1e-4
        check(f"A {kind} part 2 o(t)", (small < 1e-3).all())
    else:
        al = 0.0 if kind == "relu" else ALPHA
        check(f"A {kind} part 3 formula", np.allclose(om_, brute, atol=1e-6))
        check(f"A {kind} part 3 zero in margin", (omega(kind, a, 0.99 * np.abs(a)) == 0).all())
# ReLU at a = 0 with slope s: max(s, 1 - s) t (Comment after TV15.1)
for s_ in (0.0, 0.3, 1.0):
    tt_ = 0.7
    br = np.abs(np.maximum(UG * tt_, 0) - s_ * UG * tt_).max()
    check("A relu at 0", abs(br - max(s_, 1 - s_) * tt_) < 1e-9)

# ---------- B: Lemma TV15.2 ----------
zs = sp.Matrix(sp.symbols("z1:4"))
vs = sp.Matrix(sp.symbols("v1:4"))
es, ts = sp.symbols("eps t", positive=True)
nu_ = lambda w: sp.sqrt((w.T * w)[0] + es)
d2 = sp.diff((zs + ts * vs) / nu_(zs + ts * vs), ts, 2).subs(ts, 0)
cz = (zs.T * vs)[0]
form = (-2 * cz * vs - (vs.T * vs)[0] * zs) / nu_(zs) ** 3 + 3 * cz ** 2 * zs / nu_(zs) ** 5
check("B sympy D2 formula", sp.simplify(d2 - form) == sp.zeros(3, 1))
cc, qq = sp.symbols("c q", real=True)
check("B sympy identity", sp.expand(4 * cc ** 2 + (3 * cc ** 2 - 1) ** 2 * qq - 4 * cc ** 2 * (3 * cc ** 2 - 1)
                                    - (sp.Rational(4, 3) - (4 - 3 * qq) * (3 * cc ** 2 - 1) ** 2 / 3)) == 0)
rho, th_ = np.meshgrid(np.linspace(0, 1, 1001), np.linspace(0, np.pi, 2001))
c_ = rho * np.cos(th_)
g_ = np.hypot(-2 * c_ * np.cos(th_) - rho + 3 * c_ ** 2 * rho, -2 * c_ * np.sin(th_))
check("B sup 2/sqrt3", g_.max() <= 2 / np.sqrt(3) + 1e-12 and g_.max() > 2 / np.sqrt(3) - 1e-4, f"{g_.max()}")
for _ in range(2000):
    eps = float(rng.choice([0.0, 1e-3, 0.5, 4.0]))
    z = rng.normal(size=d) * rng.uniform(0.1, 3)
    de = rng.normal(size=d) * rng.uniform(1e-3, 2)
    beta_ = np.linalg.norm(de) * rng.uniform(1, 1.5)
    m = np.sqrt(max(np.linalg.norm(z) - beta_, 0) ** 2 + eps)
    if m <= 1e-6:
        continue
    n0, n1 = nrm(z[None], eps, True)[0], nrm((z + de)[None], eps, True)[0]
    check("B part 3 first", np.linalg.norm(n1 - n0) <= beta_ / m * (1 + 1e-9))
    check("B part 3 second", np.linalg.norm(n1 - n0 - Dn(z[None], eps)[0] @ de) <= beta_ ** 2 / (np.sqrt(3) * m ** 2)
          * (1 + 1e-9) + 1e-14)
    check("B Dn op norm", np.linalg.norm(Dn(z[None], eps)[0], 2) <= 1 / np.sqrt(z @ z + eps) * (1 + 1e-12))

# ---------- C, D: Lemma TV13.1 and Theorem TV15 ----------
gl_x, gl_w = np.polynomial.legendre.leggauss(30)
gl_x, gl_w = (gl_x + 1) / 2, gl_w / 2
ratios = {}
for norm in (False, True):
    for kind in ACT:
        for inst in range(N_INST):
            net = make_net(norm, kind)
            PD, Pi = rand_D()
            p = rng.dirichlet(np.ones(NX))
            hs, ns, As, yN = forward_N(net)
            cut = make_cut(net, Pi, noise=float(rng.choice([0.0, 0.05, 0.3])))
            hc, yB = forward_cut(net, cut, Pi)
            lk, om = leaks_omega(net, cut, Pi, hs, ns, yN)
            true = (yB - yN) @ Pperp.T
            # C: exact identity with secant matrices and difference-quotient slopes
            f = ACT[kind][0]
            dl = np.zeros((NX, d))
            for l in range(L):
                z = hs[l] @ Pi.T
                dlt = hc[l] - z
                M = sum(w_ * Dn(z + x_ * dlt, cut["eps"][l]) for x_, w_ in zip(gl_x, gl_w)) if norm else \
                    np.repeat(np.eye(d)[None], NX, 0)
                a = As[l]
                ap = nrm(hc[l], cut["eps"][l], norm) @ cut["R"][l].T + cut["b"][l]
                with np.errstate(invalid="ignore", divide="ignore"):
                    s = np.where(np.abs(ap - a) > 1e-14, (f(ap) - f(a)) / (ap - a), 0.0)
                dl = dl + (((np.einsum("xd,xde->xe", dl, M) @ cut["R"][l].T + lk[l]) * s) @ (Pi @ net["W"][l]).T)
            zL = hs[L] @ Pi.T
            dlt = hc[L] - zL
            M = sum(w_ * Dn(zL + x_ * dlt, cut["eps"][L]) for x_, w_ in zip(gl_x, gl_w)) if norm else \
                np.repeat(np.eye(d)[None], NX, 0)
            ident = om + np.einsum("xd,xde->xe", dl, M) @ (Pperp @ cut["U"]).T
            scale = 1 + np.abs(true).max()
            check(f"C {norm} {kind} identity", np.abs(ident - true).max() < 1e-7 * scale,
                  f"{np.abs(ident - true).max():.2e}")
            # D: Theorem TV15
            Lam, rem, _ = tv15(net, cut, Pi, hs, As, lk, om)
            err = np.linalg.norm(true - Lam, axis=1)
            check(f"D {norm} {kind} pointwise", (err <= rem * (1 + 1e-9) + 1e-10).all(),
                  f"{(err - rem).max():.2e}")
            Lt, remt, _ = tv15(net, cut, Pi, hs, As, lk, om, tight=True)
            check(f"D2 {norm} {kind} pointwise (part 2)", (np.linalg.norm(true - Lt, axis=1) <= remt * (1 + 1e-9) + 1e-10).all())
            check(f"D2 {norm} {kind} part 2 not looser", (remt <= rem * (1 + 1e-12) + 1e-15).all())
            DP = kl(p, yN, yB)
            bound = 0.25 * (wnorm(p, np.linalg.norm(Lam, axis=1)) + wnorm(p, rem)) ** 2
            check(f"D {norm} {kind} D_P bound", DP <= bound * (1 + 1e-9) + 1e-12)
            ratios.setdefault((norm, kind), []).append(bound / max(DP, 1e-300))

# ---------- E: Corollary TV15.3 ----------
for norm in (False, True):
    for kind in ACT:
        for inst in range(max(N_INST // 4, 3)):
            net = make_net(norm, kind)
            PD, Pi = rand_D()
            hs, ns, As, yN = forward_N(net)
            cut = make_cut(net, Pi, noise=0.05)
            lk, om = leaks_omega(net, cut, Pi, hs, ns, yN)
            Lam1, rem1, _ = tv15(net, cut, Pi, hs, As, lk, om, 1.0)
            if not np.isfinite(rem1).all():
                continue
            if kind in ("tanh", "gelu"):
                _, Rbar, _ = tv15(net, cut, Pi, hs, As, lk, om, 1.0, smooth_bar=True)
            for tsc in (0.5, 0.1, 0.01, 1e-3):
                Lt, rt, wt = tv15(net, cut, Pi, hs, As, lk, om, tsc)
                check(f"E {norm} {kind} linear Lambda", np.allclose(Lt, tsc * Lam1, rtol=1e-9, atol=1e-12))
                if kind in ("tanh", "gelu"):
                    check(f"E {norm} {kind} rem <= t^2 Rbar", (rt <= tsc ** 2 * Rbar * (1 + 1e-6) + 1e-14).all())
            if kind in ("relu", "leaky"):
                # t_1 = mu* / tau*, with tau from the t = 1 recursion (Corollary TV15.3 part 3)
                mu = min(np.abs(As[l]).min() for l in range(L))
                _, _, w1 = tv15(net, cut, Pi, hs, As, lk, om, 1.0)
                # tau*: recompute the windows at t = 1
                tau = 0.0
                beta = np.zeros(NX)
                for l in range(L):
                    z = hs[l] @ Pi.T
                    if norm:
                        m = np.sqrt(np.maximum(np.linalg.norm(z, axis=1) - beta, 0) ** 2 + cut["eps"][l])
                        kap = 1 / m
                    else:
                        kap = np.ones(NX)
                    tt = np.linalg.norm(cut["R"][l], axis=1)[None] * (kap * beta)[:, None] + np.abs(lk[l])
                    tau = max(tau, tt.max())
                    beta = beta + ACT[kind][2] * np.linalg.norm(Pi @ net["W"][l], 2) * (
                        np.linalg.norm(cut["R"][l], 2) * kap * beta + np.linalg.norm(lk[l], axis=1))
                t1 = min(1.0, mu / tau) if tau > 0 else 1.0
                _, rt, wt = tv15(net, cut, Pi, hs, As, lk, om, 0.999 * t1)
                check(f"E {norm} {kind} activation part 0 below t1", (wt == 0).all())
                if not norm:
                    check(f"E {norm} {kind} rem 0 below t1", (rt == 0).all())
                notes.append(f"{kind} norm={norm}: margin {mu:.2e}, t1 {t1:.2e}")

# ---------- F: Lemma TV16 (cuts and bounds at an angle) ----------
def orth_basis(P):
    u, sv, _ = np.linalg.svd(P)
    return u[:, sv > 0.5]


def oblique(Kb, Db):
    """Projection onto span(Kb) along span(Db)."""
    Bm = np.concatenate([Kb, Db], 1)
    Bi = np.linalg.inv(Bm)
    return Kb @ Bi[:Kb.shape[1]]


def forward_C(net, Cm, Rr, br, Ur, cyr, epsc):
    """A cut in R^(d') coordinates with any C (kernel D): first write C e, writes C W_l, C c_l."""
    f = ACT[net["act"]][0]
    z = net["e"] @ Cm.T
    reads = []
    for l in range(L):
        a = nrm(z, epsc[l], net["norm"]) @ Rr[l].T + br[l]
        reads.append(a)
        z = z + (f(a) @ net["W"][l].T + net["c"][l]) @ Cm.T
    return reads, nrm(z, epsc[L], net["norm"]) @ Ur.T + cyr


def tv6_bound(net, cut, Pi, QA, hs, ns, p):
    """Theorem TV6 part 1 (part 2 without norms) with a direct decomposition QA (Lemma TV16 part 5); logit bias = N's."""
    norm, Lip = net["norm"], ACT[net["act"]][2]
    lk, _ = leaks_omega(net, cut, Pi, hs, ns, np.zeros((NX, C)))
    m = len(QA)
    PU = Pperp @ cut["U"]
    rho = [float(np.sqrt((np.linalg.norm(hs[l] @ Pi.T, axis=1) ** 2 + cut["eps"][l]).min())) if norm else 1.0
           for l in range(L + 1)]
    T, lam = [], []
    for l in range(L):
        PW = Pi @ net["W"][l]
        aw = np.array([[np.linalg.norm(QA[A] @ PW[:, i]) for i in range(q)] for A in range(m)])   # m x q
        br = np.array([[np.linalg.norm(QA[B].T @ cut["R"][l][i]) for i in range(q)] for B in range(m)])
        rn = np.linalg.norm(cut["R"][l], axis=1)
        T.append(Lip / rho[l] * aw @ ((br + rn[None]) if norm else br).T)
        lam.append(Lip * np.abs(lk[l]) @ aw.T)                       # x by area
    if norm:
        u = np.array([np.linalg.norm(PU @ QA[A], 2) + np.linalg.norm(PU, 2) for A in range(m)]) / rho[L]
    else:
        u = np.array([np.linalg.norm(PU @ QA[A], 2) for A in range(m)])
    yN = forward_N(net)[3]
    om = (nrm(hs[L] @ Pi.T, cut["eps"][L], norm) @ cut["U"].T + net["cy"] - yN) @ Pperp.T
    tot = wnorm(p, np.linalg.norm(om, axis=1))
    for l in range(L):
        Ml = np.eye(m)
        for k in range(l + 1, L):
            Ml = (np.eye(m) + T[k]) @ Ml
        tot += float(u @ Ml @ np.sqrt((p[:, None] * lam[l] ** 2).sum(0)))
    return 0.25 * tot ** 2


for norm in (False, True):
    for kind in ACT:
        for inst in range(max(N_INST // 2, 4)):
            net = make_net(norm, kind)
            k = 3
            PD, Pi = rand_D(k)
            Db = orth_basis(PD)
            B = orth_basis(Pi)
            dp = d - k
            # part 1
            Cm = rng.normal(size=(dp, d)) @ Pi
            G_ = Cm @ B
            check(f"F1 {norm} {kind}", np.allclose(Cm, G_ @ B.T @ Pi, atol=1e-10) and abs(np.linalg.det(G_)) > 1e-9)
            # part 2 / 3
            Rr = [rng.normal(size=(q, dp)) / np.sqrt(dp) * (3 if norm else 1) for _ in range(L)]
            br_ = [rng.normal(size=q) * .3 for _ in range(L)]
            Ur = rng.normal(size=(C, dp))
            epsc = [net["eps"][l] + 0.3 for l in range(L + 1)]
            r1, y1 = forward_C(net, Cm, Rr, br_, Ur, net["cy"], epsc)
            r2, y2 = forward_C(net, B.T @ Pi, [R @ G_ for R in Rr], br_, Ur @ G_, net["cy"], epsc)
            if not norm:
                check(f"F2 {norm} {kind}", np.allclose(y1, y2, atol=1e-9 * (1 + np.abs(y1).max())))
            else:
                z = net["e"] @ (B.T @ Pi).T
                s1 = np.sqrt(((z * z).sum(1) + epsc[0]) / (((z @ G_.T) ** 2).sum(1) + epsc[0]))
                check(f"F3 {norm} {kind} scalar", np.allclose(r1[0] - br_[0], s1[:, None] * (r2[0] - br_[0]), atol=1e-9))
            # part 4: the oblique cut, Theorem TV15 with Q
            Kb = np.linalg.qr(rng.normal(size=(d, dp)))[0]
            Q = oblique(Kb, Db)
            PK = Kb @ Kb.T
            hs, ns, As, yN = forward_N(net)
            cut = make_cut(net, PK, noise=float(rng.choice([0.0, 0.05, 0.3])))
            hc, yB = forward_cut(net, cut, Q)
            # the R^d-coordinates oblique cut equals the R^(d') one with C = B_K^T Q
            _, yC = forward_C(net, Kb.T @ Q, [R @ Kb for R in cut["R"]], cut["b"], cut["U"] @ Kb, cut["cy"], cut["eps"])
            check(f"F4 {norm} {kind} TV0 for Q", np.allclose(yB, yC, atol=1e-9 * (1 + np.abs(yB).max())))
            lk, om = leaks_omega(net, cut, Q, hs, ns, yN)
            Lam, rem, _ = tv15(net, cut, Q, hs, As, lk, om)
            err = np.linalg.norm((yB - yN) @ Pperp.T - Lam, axis=1)
            check(f"F4 {norm} {kind} TV15 oblique", (err <= rem * (1 + 1e-9) + 1e-10).all(), f"{(err - rem).max():.2e}")
            # part 5: TV6 with a random direct decomposition, both kinds of cut (logit bias = N's)
            Bm = rng.normal(size=(d, d))
            Bi = np.linalg.inv(Bm)
            sizes = [3, 3, 4]
            idx = np.split(np.arange(d), np.cumsum(sizes)[:-1])
            QA = [Bm[:, I_] @ Bi[I_] for I_ in idx]
            check("F5 decomposition", np.allclose(sum(QA), np.eye(d)))
            p = rng.dirichlet(np.ones(NX))
            for proj_, read_ in ((Pi, Pi), (Q, PK)):
                cut = make_cut(net, read_, noise=float(rng.choice([0.0, 0.05, 0.3])))
                cut["cy"] = net["cy"]
                hc, yB = forward_cut(net, cut, proj_)
                bnd = tv6_bound(net, cut, proj_, QA, hs, ns, p)
                check(f"F5 {norm} {kind} TV6 direct decomposition", kl(p, yN, yB) <= bnd * (1 + 1e-9) + 1e-12,
                      f"{kl(p, yN, yB):.3e} > {bnd:.3e}")

# ---------- G: Proposition TV17 ----------
for kind in ACT:
    for inst in range(5):
        net = make_net(True, kind)
        v = rng.normal(size=d); v /= np.linalg.norm(v)
        vp = v + 0.8 * rng.normal(size=d) / np.sqrt(d)          # K = vp perp, at an angle to D = span(v)
        Kb = orth_basis(np.eye(d) - np.outer(vp, vp) / (vp @ vp))
        Q = oblique(Kb, v[:, None])
        PKv = orth_basis((np.eye(d) - np.outer(vp, vp) / (vp @ vp)))   # K
        # writes inside K and orthogonal to v keep e_Q constant; reads and logits kill D
        Wsp = orth_basis(Kb @ Kb.T - np.outer(Kb @ (Kb.T @ v), Kb @ (Kb.T @ v)) / max((Kb.T @ v) @ (Kb.T @ v), 1e-12))
        Pw = Wsp @ Wsp.T
        net["W"] = [Pw @ W for W in net["W"]]
        net["c"] = [Pw @ c for c in net["c"]]
        Pv = np.eye(d) - np.outer(v, v)
        net["R"] = [R @ Pv for R in net["R"]]
        net["U"] = net["U"] @ Pv
        cc_ = 0.7
        k0 = net["e"] @ (Kb @ Kb.T).T
        kv = k0 @ v
        s = -kv + np.sqrt(kv ** 2 + cc_)
        net["e"] = k0 + s[:, None] * v[None]
        hs, ns, As, yN = forward_N(net)
        eq = [(hs[l] ** 2).sum(1) - ((hs[l] @ Q.T) ** 2).sum(1) for l in range(L + 1)]
        check(f"G1 {kind} e_Q constant", all(np.allclose(e_, cc_, atol=1e-9) for e_ in eq))
        _, yC = forward_C(net, Kb.T @ Q, [R @ Kb for R in net["R"]], net["b"], net["U"] @ Kb, net["cy"],
                          [e_ + cc_ for e_ in net["eps"]])
        check(f"G1 {kind} oblique cut exact", np.allclose(yC, yN, atol=1e-9 * (1 + np.abs(yN).max())))
# part 2: the example
for t_ in (0.3, 1.0, 1.4):
    u_ = np.array([np.cos(t_), np.sin(t_), 0.0])
    pts = [a * np.array([1.0, 0, 0]) + b * u_ + np.array([0, 0, 1.0]) for a in (1, -1) for b in (1, -1)]
    H_ = np.array(pts)
    Bm = np.stack([np.array([1.0, 0, 0]), u_, np.array([0, 0, 1.0])], 1)
    Bi = np.linalg.inv(Bm)
    Q1, Q2 = Bm[:, :1] @ Bi[:1], Bm[:, 1:2] @ Bi[1:2]
    check("G2 per-area energies constant", np.allclose(((H_ @ Q1.T) ** 2).sum(1), 1) and
          np.allclose(((H_ @ Q2.T) ** 2).sum(1), 1))
    yN_ = np.stack([H_[:, 2] / np.sqrt((H_ ** 2).sum(1) + 1), np.zeros(4)], 1)
    pw = np.full(4, 0.25)
    best = min(kl(pw, yN_, np.tile(np.array([g, 0.0]), (4, 1))) for g in np.linspace(-3, 3, 60001))
    check("G2 every cut lossy", best > 1e-6, f"{best:.2e}")
    notes.append(f"TV17 example t={t_}: least D_P of a constant logit vector {best:.3e}")
    # second example: the orthogonal (DM13) per-area energies are constant
    H2 = np.array([[x, (s - x * np.cos(t_)) / np.sin(t_), 1.0] for x in (1, -1) for s in (1, -1)])
    check("G2b orthogonal per-area energies constant", np.allclose(H2[:, 0] ** 2, 1) and np.allclose((H2 @ u_) ** 2, 1))
    PDn = ((H2[:, :2]) ** 2).sum(1)
    check("G2b |P_D h|^2 takes two values", np.ptp(PDn) > 1e-6)
    yN2 = np.stack([H2[:, 2] / np.sqrt((H2 ** 2).sum(1) + 1), np.zeros(4)], 1)
    best2 = min(kl(pw, yN2, np.tile(np.array([g, 0.0]), (4, 1))) for g in np.linspace(-3, 3, 60001))
    check("G2b every cut lossy", best2 > 1e-6, f"{best2:.2e}")

FAM = []
# the family N_t: D-content of N scaled by t; projection reads; the cut does not depend on t
for norm in (False, True):
    for kind in ("relu", "tanh", "gelu"):
        for inst in range(3):
            base = make_net(norm, kind)
            PD, Pi = rand_D()
            for tsc in (1.0, 0.1, 1e-2, 1e-3, 1e-4):
                net = dict(base)
                net["e"] = base["e"] @ Pi.T + tsc * base["e"] @ PD.T
                net["W"] = [Pi @ W + tsc * PD @ W for W in base["W"]]
                net["c"] = [Pi @ c + tsc * PD @ c for c in base["c"]]
                hs, ns, As, yN = forward_N(net)
                cut = dict(R=[R @ Pi for R in net["R"]], b=list(net["b"]), U=net["U"] @ Pi, cy=net["cy"],
                           eps=list(net["eps"]))
                hc, yB = forward_cut(net, cut, Pi)
                lk, om = leaks_omega(net, cut, Pi, hs, ns, yN)
                Lam, rem, wt = tv15(net, cut, Pi, hs, As, lk, om)
                true = (yB - yN) @ Pperp.T
                err = np.linalg.norm(true - Lam, axis=1)
                check(f"E family {norm} {kind} pointwise t={tsc}", (err <= rem * (1 + 1e-9) + 1e-12).all())
                if kind == "relu" and not norm and (wt == 0).all():
                    check(f"E family relu exact t={tsc}", err.max() < 1e-10 * (1 + np.abs(true).max()),
                          f"{err.max():.2e}")
                    notes.append(f"family relu no-norm inst {inst}: exact at t={tsc} (err {err.max():.1e}, "
                                 f"|change| {np.abs(true).max():.1e})")
                if inst == 0 and kind in ("gelu", "relu"):
                    pf = np.full(NX, 1 / NX)
                    DPf = kl(pf, yN, yB)
                    b15 = 0.25 * (wnorm(pf, np.linalg.norm(Lam, axis=1)) + wnorm(pf, rem)) ** 2
                    b6 = tv6_bound(net, cut, Pi, [PD, Pi], hs, ns, pf)
                    FAM.append(f"{kind} norm={norm} t={tsc:g}: TV15/D_P {b15 / DPf:.3g}, TV6/D_P {b6 / DPf:.3g}")
                if tsc == 1e-4:
                    nt = np.linalg.norm(true, axis=1).max()
                    notes.append(f"family {kind} norm={norm} t=1e-4: |change| {nt:.2e}, rem max {rem.max():.2e}")

for k, v in ratios.items():
    notes.append(f"D ratio bound/D_P norm={k[0]} {k[1]}: median {np.median(v):.3g}")
print("\n".join(notes[-40:]))
print("\n".join(FAM))
print("checks run:", COUNT)
print(f"tv_norm: {len(fails)} failures")
for f_ in fails[:30]:
    print("FAIL", f_)
