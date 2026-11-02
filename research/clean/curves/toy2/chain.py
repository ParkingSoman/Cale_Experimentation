"""Toy gated chains for COARSE-THEORY.md Part B.4 (library; the scripts that use it carry the '# serves:' lines).

Polar grid: inputs on JI rings x BN angle bins, positions on JP rings x BN bins. The group Z_N (N = 8) acts by a roll
of SH = BN / N bins, exactly. Shapes are rendered in the plane and sampled at the polar grid points, so a rotation by
2 pi / N is an exact roll. Classes: (shape type s, orientation j), j in Z_N. A roll by SH bins sends class j to j + 1.

Teacher (H-gate, H-eq by construction):
  member i kernel K_i = roll of the member-0 kernel (weight tying), gates g_1..g_3 from invariant kernels (group
  average in the forward pass), circulant edges, readout H[(s, j), i] = h_s[i - j], bias constant on each orbit.
Variant 'relu1': the first family's members also pass through their own ReLU, v_1 = g_1 relu(z) (breaks H-gate at
  the first family only; the chain above stays gated-linear).
Variant 'selfgate' (H-selfgate, Part B.5): no gate path. Gate l reads the orbit mean of the layer below:
  g_1 = relu(a_1 mean_i z_i + b_1), g_{l+1} = relu(a_{l+1} mean_i v_{l,i} + b_{l+1}). Forced read space = span{1}, r = 1.
"""
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
N, BN, JI, JP = 8, 32, 5, 3
SH = BN // N
NT = 3                                   # shape types
NY = NT * N
D = JI * BN + 1
PN = JP * BN

r_in = (np.arange(JI) + 0.5) / JI
th = 2 * np.pi * np.arange(BN) / BN
GX = r_in[:, None] * np.cos(th)[None, :]
GY = r_in[:, None] * np.sin(th)[None, :]


def stroke(x0, y0, x1, y1, w=0.09):
    px, py = GX - x0, GY - y0
    dx, dy = x1 - x0, y1 - y0
    t = np.clip((px * dx + py * dy) / (dx * dx + dy * dy + 1e-12), 0, 1)
    d2 = (px - t * dx) ** 2 + (py - t * dy) ** 2
    return np.exp(-d2 / (2 * w * w))


def render(s, a, rng):
    c, sn = np.cos(a), np.sin(a)
    rot = lambda u, v: (u * c - v * sn, u * sn + v * c)
    sc = rng.uniform(0.8, 1.1)
    if s == 0:     # arrow: shaft and one barb
        img = stroke(*rot(0, 0), *rot(0.9 * sc, 0)) + stroke(*rot(0.9 * sc, 0), *rot(0.6 * sc, 0.25 * sc))
    elif s == 1:   # L corner off centre
        img = stroke(*rot(0.2, -0.3 * sc), *rot(0.8 * sc, -0.3 * sc)) + stroke(*rot(0.8 * sc, -0.3 * sc), *rot(0.8 * sc, 0.3 * sc))
    else:          # arc bulging toward angle a
        ts = np.linspace(-1.0, 1.0, 9)
        pts = [rot(0.35 + 0.35 * sc * np.cos(t), 0.6 * sc * np.sin(t)) for t in ts]
        img = sum(stroke(*pts[q], *pts[q + 1]) for q in range(len(pts) - 1))
    blob = np.exp(-((GX - rng.uniform(-.8, .8)) ** 2 + (GY - rng.uniform(-.8, .8)) ** 2) / 0.05) * rng.uniform(0, .8)
    return np.clip(img, 0, 1.2) * rng.uniform(0.7, 1.2) + blob + rng.normal(0, 0.05, img.shape)


def sample(n, seed, sym=True, jitter=0.6):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, JI, BN)); y = np.zeros(n, dtype=np.int64)
    for k in range(n):
        s, j = rng.integers(NT), rng.integers(N)
        a = 2 * np.pi * j / N + rng.uniform(-1, 1) * jitter * np.pi / N
        X[k] = render(s, a, rng); y[k] = s * N + j
    if sym:
        X = np.concatenate([np.roll(X, q * SH, axis=-1) for q in range(N)])
        y = np.concatenate([(y // N) * N + (y % N + q) % N for q in range(N)])
    Xt = np.concatenate([X.reshape(len(X), -1), np.ones((len(X), 1))], axis=1)
    return torch.tensor(Xt), torch.tensor(y)


def roll_kernel(K, q):
    """gamma^q K(gamma^-q p) for kernels of shape (..., JP, BN, D): roll positions and input bins by q * SH."""
    # positions bin axis is dim -2 of K (shape ..., JP, BN, D); input bin axis is the last of pix
    pix = K[..., :-1].reshape(K.shape[:-1] + (JI, BN))
    pix = torch.roll(pix, shifts=q * SH, dims=-1)
    bias = K[..., -1:]
    out = torch.cat([pix.reshape(K.shape[:-1] + (JI * BN,)), bias], dim=-1)
    return torch.roll(out, shifts=q * SH, dims=-2)


class Teacher(torch.nn.Module):
    def __init__(self, seed=0, variant="gated"):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.variant = variant
        self.K0 = torch.nn.Parameter(torch.randn(JP, BN, D, generator=g) * 0.05)
        self.Kg = torch.nn.Parameter(torch.randn(3, JP, BN, D, generator=g) * 0.05)
        with torch.no_grad():
            self.Kg[..., -1] = 0.5
        self.w = torch.nn.Parameter(torch.randn(2, N, generator=g) * 0.5)
        self.h = torch.nn.Parameter(torch.randn(NT, N, generator=g) * 0.3)
        self.cb = torch.nn.Parameter(torch.zeros(NT))
        if variant == "selfgate":
            self.ga = torch.nn.Parameter(torch.full((3,), 2.0))
            self.gb = torch.nn.Parameter(torch.full((3,), 1.0))

    def members(self):
        return torch.stack([roll_kernel(self.K0, i) for i in range(N)])            # (N, JP, BN, D)

    def gate_kernels(self):
        return torch.stack([roll_kernel(self.Kg, q) for q in range(N)]).mean(0)    # invariant (3, JP, BN, D)

    def edges(self):
        return [torch.stack([torch.roll(self.w[l], i) for i in range(N)]) for l in range(2)]   # circulant

    def head(self):
        H = torch.stack([torch.stack([torch.roll(self.h[s], j) for j in range(N)]) for s in range(NT)])
        return H.reshape(NY, N), self.cb.repeat_interleave(N)

    def gates(self, X):
        if self.variant == "selfgate":
            return self.forward(X, parts=True)[3]
        Kg = self.gate_kernels().reshape(3, PN, D)
        return torch.relu(torch.einsum("sd,lpd->lsp", X, Kg))                       # (3, S, PN)

    def forward(self, X, parts=False):
        z = torch.einsum("sd,ipd->sip", X, self.members().reshape(N, PN, D))
        w1, w2 = self.edges()
        if self.variant == "selfgate":
            g0 = torch.relu(self.ga[0] * z.mean(1) + self.gb[0])
            v1 = g0[:, None] * z
            g1 = torch.relu(self.ga[1] * v1.mean(1) + self.gb[1])
            v2 = g1[:, None] * torch.einsum("ij,sjp->sip", w1, v1)
            g2 = torch.relu(self.ga[2] * v2.mean(1) + self.gb[2])
            v3 = g2[:, None] * torch.einsum("ij,sjp->sip", w2, v2)
            g = torch.stack([g0, g1, g2])
        else:
            g = self.gates(X)
            v1 = g[0][:, None] * (torch.relu(z) if self.variant == "relu1" else z)
            v2 = g[1][:, None] * torch.einsum("ij,sjp->sip", w1, v1)
            v3 = g[2][:, None] * torch.einsum("ij,sjp->sip", w2, v2)
        H, c = self.head()
        y = torch.einsum("yi,sip->sy", H, v3) / PN + c
        if parts:
            G = g[0] * g[1] * g[2]
            uhat = torch.einsum("sip,sp->si", torch.relu(z) if self.variant == "relu1" else z, G) / PN
            return y, uhat, (v1, v2, v3), g, z
        return y


class Student(torch.nn.Module):
    """A build of widths (k1, k2, k3): same gate path (frozen, passed in), free value weights."""

    def __init__(self, k1, k2, k3, seed=0, relu1=False, scale=0.05):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.relu1 = relu1
        self.K = torch.nn.Parameter(torch.randn(k1, PN, D, generator=g) * scale)
        self.w1 = torch.nn.Parameter(torch.randn(k2, k1, generator=g) / np.sqrt(k1))
        self.w2 = torch.nn.Parameter(torch.randn(k3, k2, generator=g) / np.sqrt(k2))
        self.H = torch.nn.Parameter(torch.randn(NY, k3, generator=g) * 0.3)
        self.c = torch.nn.Parameter(torch.zeros(NY))

    def forward(self, X, g):
        z = torch.einsum("sd,apd->sap", X, self.K)
        c1 = g[0][:, None] * (torch.relu(z) if self.relu1 else z)
        c2 = g[1][:, None] * torch.einsum("ab,sbp->sap", self.w1, c1)
        c3 = g[2][:, None] * torch.einsum("ab,sbp->sap", self.w2, c2)
        return torch.einsum("ya,sap->sy", self.H, c3) / PN + self.c


PPERP = torch.eye(NY) - torch.ones(NY, NY) / NY


def d2(y, yb):
    return 0.5 * (((yb - y) @ PPERP) ** 2).sum(1).mean()


def fourier_basis():
    t = 2 * np.pi * np.arange(N) / N
    cols, labels = [np.ones(N) / np.sqrt(N)], [0]
    for n in range(1, N // 2):
        cols += [np.cos(n * t) * np.sqrt(2 / N), np.sin(n * t) * np.sqrt(2 / N)]
        labels += [n, n]
    cols.append(np.cos(N // 2 * t) / np.sqrt(N)); labels.append(N // 2)
    return torch.tensor(np.stack(cols, axis=1)), np.array(labels)


def formula(T, X):
    """Theorem T9 (general) and T10 (harmonic) quantities on the law 'uniform over X'."""
    with torch.no_grad():
        y, uhat, vs, g, z = T(X, parts=True)
        w1, w2 = T.edges(); H, c = T.head()
        A = H @ w2 @ w1
        Eu = uhat.mean(0); uc = uhat - Eu
        Chat = uc.T @ uc / len(X)
        yc = (y - y.mean(0)) @ PPERP
        covy = yc.T @ yc / len(X)
        mu, E = torch.linalg.eigh(covy); o = torch.argsort(mu, descending=True); mu, E = mu[o], E[:, o]
        F, labels = fourier_basis()
        ev, V = torch.linalg.eigh(Chat)
        Ch = V @ torch.diag(ev.clamp(min=0).sqrt()) @ V.T
        lam = torch.diag(F.T @ Ch @ (A.T @ PPERP @ A) @ Ch @ F)
    return dict(y=y, uhat=uhat, g=g, A=A, Eu=Eu, Chat=Chat, mu=mu, E=E, F=F, labels=labels, lam=lam, H=H, c=c,
                w=(w1, w2))


def strict_widths(fm):
    lam, labels = fm["lam"].numpy(), fm["labels"]
    modes = sorted(set(labels.tolist()), key=lambda n: -lam[labels == n][0])
    out, acc = [], []
    for n in modes:
        acc = acc + [n]
        out.append((int(sum((labels == m).sum() for m in acc)), list(acc)))
    return out


def t9_student(T, fm, k):
    """Theorem T9 part 2 as a Student (identity edges)."""
    Uk = fm["E"][:, :k]
    Lam = Uk.T @ PPERP @ fm["A"]
    S = Student(k, k, k)
    with torch.no_grad():
        S.K.copy_(torch.einsum("ai,ipd->apd", Lam, T.members().reshape(N, PN, D)))
        S.w1.copy_(torch.eye(k)); S.w2.copy_(torch.eye(k)); S.H.copy_(Uk)
        S.c.copy_(fm["c"] + (fm["A"] - Uk @ Lam) @ fm["Eu"])
    return S


def t10_student(T, fm, modes):
    """Theorem T10 part 3 as a Student (Q_S at every layer)."""
    F, labels = fm["F"], fm["labels"]
    Q = F[:, np.isin(labels, modes)]
    k = Q.shape[1]
    w1, w2 = fm["w"]
    S = Student(k, k, k)
    with torch.no_grad():
        S.K.copy_(torch.einsum("ia,ipd->apd", Q, T.members().reshape(N, PN, D)))
        S.w1.copy_(Q.T @ w1 @ Q); S.w2.copy_(Q.T @ w2 @ Q); S.H.copy_(fm["H"] @ Q)
        S.c.copy_(fm["c"] + fm["A"] @ (torch.eye(N) - Q @ Q.T) @ fm["Eu"])
    return S


class SelfStudent(torch.nn.Module):
    """A self-contained build of uniform width k for the 'selfgate' teacher (Part B.5): no gate path, gate l reads one
    of the build's own units through the teacher's gate function relu(a_l t + b_l).
    faithful=True: unit 0 is the forced read (kernel = mean of the member kernels, fixed), the edges keep it
      (row 0 of w_l is e_0^T, fixed) and every gate reads unit 0, times the fixed scale that makes it the teacher's
      read (1, 1, sigma_1 with sigma_1 the column sum of the teacher's first circulant edge: the read matrices M of
      Theorem T11 part 2). So the build is read-faithful (Definition D12) and its gates are the teacher's (Lemma T11a).
    faithful=False: every weight is free and gate l reads s_l^T (layer l - 1 units) with a free read s_l (Conjecture D6).
    """

    def __init__(self, T, k, seed=0, faithful=True, scale=0.05):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.k, self.faithful = k, faithful
        self.ga, self.gb = T.ga.detach().clone(), T.gb.detach().clone()
        w1, w2 = T.edges()
        self.sig = (float(w1[:, 0].sum()), float(w2[:, 0].sum()))
        self.register_buffer("K0", T.members().detach().mean(0).reshape(1, PN, D))
        nfree = k - 1 if faithful else k
        self.K = torch.nn.Parameter(torch.randn(nfree, PN, D, generator=g) * scale)
        self.w1 = torch.nn.Parameter(torch.randn(k, k, generator=g) / np.sqrt(k))
        self.w2 = torch.nn.Parameter(torch.randn(k, k, generator=g) / np.sqrt(k))
        self.H = torch.nn.Parameter(torch.randn(NY, k, generator=g) * 0.3)
        self.c = torch.nn.Parameter(torch.zeros(NY))
        if not faithful:
            self.s = torch.nn.Parameter(torch.randn(3, k, generator=g) / np.sqrt(k))

    def kernels(self):
        return torch.cat([self.K0, self.K]) if self.faithful else self.K

    def edges(self):
        if not self.faithful:
            return self.w1, self.w2
        out = []
        for w in (self.w1, self.w2):
            row0 = torch.zeros(1, self.k, dtype=w.dtype); row0[0, 0] = 1.0
            out.append(torch.cat([row0, w[1:]]))
        return out

    def read(self, l, v):
        return (1.0, 1.0, self.sig[0])[l] * v[:, 0] if self.faithful else torch.einsum("a,sap->sp", self.s[l], v)

    def forward(self, X, g_unused=None, parts=False):
        z = torch.einsum("sd,apd->sap", X, self.kernels())
        w1, w2 = self.edges()
        g0 = torch.relu(self.ga[0] * self.read(0, z) + self.gb[0])
        v1 = g0[:, None] * z
        g1 = torch.relu(self.ga[1] * self.read(1, v1) + self.gb[1])
        v2 = g1[:, None] * torch.einsum("ab,sbp->sap", w1, v1)
        g2 = torch.relu(self.ga[2] * self.read(2, v2) + self.gb[2])
        v3 = g2[:, None] * torch.einsum("ab,sbp->sap", w2, v2)
        y = torch.einsum("ya,sap->sy", self.H, v3) / PN + self.c
        return (y, torch.stack([g0, g1, g2])) if parts else y


def t11_quantities(fm):
    """Theorem T11 with R_0 = 1 / N (the orbit-mean read): forced feature f, C_f, C_yf, residual covariance, nu, E'."""
    uhat, A, Chat = fm["uhat"], fm["A"], fm["Chat"]
    R0 = torch.full((N, 1), 1.0 / N)
    Cf = R0.T @ Chat @ R0
    Cyf = PPERP @ A @ Chat @ R0
    covy = PPERP @ A @ Chat @ A.T @ PPERP
    Ce = covy - Cyf @ torch.linalg.solve(Cf, Cyf.T)
    nu, E = torch.linalg.eigh(Ce); o = torch.argsort(nu, descending=True)
    return dict(R0=R0, Cf=Cf, Cyf=Cyf, nu=nu[o], E=E[:, o], B=Cyf @ torch.linalg.inv(Cf))


def t11_student(T, fm, q, k):
    """Theorem T11 part 2 as a faithful SelfStudent: unit 0 = orbit mean, units 1..k-1 = U'^T (P A - B R0^T) reads."""
    Up = q["E"][:, :k - 1]
    Lp = Up.T @ (PPERP @ fm["A"] - q["B"] @ q["R0"].T)                 # (k-1, N)
    S = SelfStudent(T, k)
    with torch.no_grad():
        if k > 1:
            S.K.copy_(torch.einsum("ai,ipd->apd", Lp, T.members().reshape(N, PN, D)))
        S.w1.copy_(torch.eye(k)); S.w2.copy_(torch.eye(k))
        Hp = torch.cat([q["B"], Up], dim=1)
        S.H.copy_(Hp)
        Lam = torch.cat([q["R0"].T, Lp])
        Heff = torch.cat([q["B"], Up], dim=1)
        S.c.copy_(fm["c"] + (fm["A"] - Heff @ Lam) @ fm["Eu"])
    return S


def t11_1_build(T, fm, q, k):
    """Corollary T11.1 for the toy: widths (k + 1, k + 1, k). Layers 1, 2 carry the orbit mean and the T9 read; the last
    edge drops the forced unit. Returns a function X -> logits (self-contained)."""
    Uk = fm["E"][:, :k]
    L9 = Uk.T @ PPERP @ fm["A"]
    K9 = torch.einsum("ai,ipd->apd", L9, T.members().reshape(N, PN, D))
    K0 = T.members().mean(0).reshape(1, PN, D)
    w1, w2 = T.edges(); s1 = float(w1[:, 0].sum())
    cb = fm["c"] + (fm["A"] - Uk @ L9) @ fm["Eu"]

    def f(X):
        z = torch.einsum("sd,apd->sap", X, torch.cat([K0, K9]))
        g0 = torch.relu(T.ga[0] * z[:, 0] + T.gb[0]); v1 = g0[:, None] * z
        g1 = torch.relu(T.ga[1] * v1[:, 0] + T.gb[1]); v2 = g1[:, None] * v1
        g2 = torch.relu(T.ga[2] * s1 * v2[:, 0] + T.gb[2]); v3 = g2[:, None] * v2[:, 1:]
        return torch.einsum("ya,sap->sy", Uk, v3) / PN + cb, torch.stack([g0, g1, g2])
    return f
