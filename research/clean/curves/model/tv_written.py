# serves: TV19
"""Exact checks of Part B.V.7 (COARSE-THEORY.md): Lemma TV19, Proposition TV19.1, Corollary TV19.3, Proposition TV19.4.
Random pure-stream networks (d = 16, three blocks of 8 neurons, C = 4 logits), tanh and ReLU, both norm forms.
  A  TV19 / TV19.3: a unit build of a zoom (every neuron in a group, or some kept), first write of rank r; the stream
     on P lies in E_N + new writes, and the cut to that subspace has the same inside values and logits (1e-9)
  B  control: a full-rank first write gives a written span of dimension d (nothing is free)
  C  TV19.1: no norm, reads planted to miss part of the written span: the quotient network of width
     dim V - dim(V cap Z) matches; norm form with a constant-energy dropped part (eps + c) matches
  D  TV19.4: the example's differences are independent, and the best width-1 fit found has D_P > 0 (the fit lets z
     take any value at each point, a wider class than Bc'_1, whose z = q^T x; so its infimum bounds Bc'_1's below)
  E  TV19.2, the lift: a random narrow unit build of Definition DV6 (an isometry Q whose range holds E_N, kept
     neurons Q^T r / Q^T w, free new neurons) and its lift (new neurons Q r', Q w') agree at every point (1e-9)
Run: python model/tv_written.py [instances]"""
import sys
import numpy as np

rng = np.random.default_rng(1931)
N_INST = int(sys.argv[1]) if len(sys.argv) > 1 else 20
d, C, NX, L, q, DX = 16, 4, 80, 3, 8, 5
Pperp = np.eye(C) - np.ones((C, C)) / C
fails, COUNT = [], {}
ACT = {"tanh": np.tanh, "relu": lambda a: np.maximum(a, 0)}


def check(name, ok, info=""):
    COUNT[name.split()[0]] = COUNT.get(name.split()[0], 0) + 1
    if not ok:
        fails.append(f"{name} {info}")


def nrm(h, eps, on):
    return h / np.sqrt((h * h).sum(1, keepdims=True) + eps) if on else h


def forward(net, X):
    """net: dict with e (function), blocks [(R, b, W, c)], U, cy, eps list, norm flag, act. Returns hs, gs, y."""
    h = net["e"](X)
    hs, gs = [h], []
    for l, (R, b, W, c) in enumerate(net["blocks"]):
        g = ACT[net["act"]](nrm(h, net["eps"][l], net["norm"]) @ R.T + b)
        h = h + g @ W.T + c
        hs.append(h)
        gs.append(g)
    y = nrm(h, net["eps"][L], net["norm"]) @ net["U"].T + net["cy"]
    return hs, gs, y


def span(M, tol=1e-9):
    """orthonormal basis (columns) of the column span of M."""
    if M.shape[1] == 0:
        return np.zeros((M.shape[0], 0))
    u, s, _ = np.linalg.svd(M, full_matrices=False)
    return u[:, s > tol * max(1.0, s[0])]


def make_build(norm, act, r, kept_frac):
    """N-like network: first write e(x) = A x with rank r plus a bias, blocks of q neurons; neurons are 'kept' (fixed
    write) or 'new' (a group replaced by fewer new neurons). Returns net, E_N basis, list of new write vectors."""
    A = rng.normal(size=(d, DX))
    if r < DX:
        A = A @ np.linalg.qr(rng.normal(size=(DX, DX)))[0][:, :r] @ rng.normal(size=(r, DX))
    e0 = rng.normal(size=d) * 0.3
    blocks, kept_w, new_w = [], [], []
    for l in range(L):
        nk = int(round(q * kept_frac))
        nn = rng.integers(1, 3)          # the group of q - nk neurons is replaced by nn new neurons
        R = rng.normal(size=(nk + nn, d)) / np.sqrt(d)
        b = rng.normal(size=nk + nn) * 0.2
        W = rng.normal(size=(d, nk + nn)) / np.sqrt(q)
        c = rng.normal(size=d) * 0.1
        kept_w += [W[:, i] for i in range(nk)]
        new_w += [W[:, i] for i in range(nk, nk + nn)]
        blocks.append((R, b, W, c))
    net = dict(e=lambda X: X @ A.T + e0, blocks=blocks, U=rng.normal(size=(C, d)), cy=rng.normal(size=C) * 0.1,
               eps=[0.5] * (L + 1), norm=norm, act=act)
    X = rng.normal(size=(NX, DX))
    EN = span(np.column_stack([(X @ A.T + e0).T] + [blk[3][:, None] for blk in blocks] +
                              ([np.column_stack(kept_w)] if kept_w else [])))
    return net, X, EN, new_w


def cut_net(net, B, epsadd=None):
    """the network of Lemma TV19 / TV19.1 on the columns of B (orthonormal)."""
    e = net["e"]
    blocks = [(R @ B, b, B.T @ W, B.T @ c) for (R, b, W, c) in net["blocks"]]
    eps = net["eps"] if epsadd is None else [x + a for x, a in zip(net["eps"], epsadd)]
    return dict(e=lambda X: e(X) @ B, blocks=blocks, U=net["U"] @ B, cy=net["cy"], eps=eps, norm=net["norm"],
                act=net["act"])


def same(net, net2, X, tag):
    hs, gs, y = forward(net, X)
    hs2, gs2, y2 = forward(net2, X)
    dg = max(np.abs(g - g2).max() for g, g2 in zip(gs, gs2))
    dy = np.abs((y - y2) @ Pperp).max()
    check(tag, dg < 1e-9 and dy < 1e-9, f"dg={dg:.2e} dy={dy:.2e}")


for inst in range(N_INST):
    for norm in (False, True):
        for act in ("tanh", "relu"):
            # A: low-rank first write, every block's group replaced by 1-2 new neurons, some kept neurons
            for kept_frac in (0.0, 0.25):
                net, X, EN, new_w = make_build(norm, act, r=2, kept_frac=kept_frac)
                hs, _, _ = forward(net, X)
                Vb = span(np.column_stack([h.T for h in hs]))
                K = span(np.column_stack([EN] + [w[:, None] for w in new_w]))
                # the written span lies in E_N + new writes (Corollary TV19.3's inclusion)
                res = np.abs(Vb - K @ (K.T @ Vb)).max()
                check("A-incl", res < 1e-9, f"res={res:.2e}")
                check("A-count", Vb.shape[1] <= EN.shape[1] + len(new_w))
                same(net, cut_net(net, Vb), X, "A-cutV")
                same(net, cut_net(net, K), X, "A-cutK")
                if kept_frac == 0.0:
                    check("A-narrow", K.shape[1] < d, f"dimK={K.shape[1]}")
            # B: control, full-rank first write
            net, X, EN, new_w = make_build(norm, act, r=DX, kept_frac=0.0)
            net["e"] = (lambda Afull: (lambda X: np.column_stack([X, X ** 2, np.sin(X), np.cos(X)])[:, :d] @ Afull.T))(
                rng.normal(size=(d, d)))
            hs, _, _ = forward(net, X)
            check("B-full", span(np.column_stack([h.T for h in hs])).shape[1] == d)

    # C: read side. Plant a subspace Zp of the written span that no reader reads.
    for act in ("tanh", "relu"):
        for norm in (False, True):
            net, X, EN, new_w = make_build(norm, act, r=2, kept_frac=0.0)
            z0 = np.linalg.qr(rng.normal(size=(d, d)))[0][:, :1]          # the planted unread direction
            Pz = np.eye(d) - z0 @ z0.T
            e_old = net["e"]
            if norm:   # constant energy on z0: the first write carries +-0.7 z0 (sign of x_0), nobody writes into z0
                # (a constant 0.7 z0 would merge with the constant part of the first write and leave V cap Z = 0)
                net["e"] = lambda X, e_old=e_old: e_old(X) @ Pz + 0.7 * np.sign(X[:, :1]) * z0[:, 0]
                net["blocks"] = [(R @ Pz, b, Pz @ W, Pz @ c) for (R, b, W, c) in net["blocks"]]
            else:      # anything may be written into z0, nobody reads it
                net["blocks"] = [(R @ Pz, b, W, c) for (R, b, W, c) in net["blocks"]]
                net["e"] = lambda X, e_old=e_old: e_old(X) + np.sin(X[:, :1]) * z0[:, 0]
            net["U"] = net["U"] @ Pz
            hs, _, _ = forward(net, X)
            V = span(np.column_stack([h.T for h in hs]))
            if np.abs(z0 - V @ (V.T @ z0)).max() > 1e-9:
                # with a norm, z0 lies in V only if the stream values satisfy no affine relation; else the
                # proposition has nothing to drop here (V cap span(z0) = 0): count and skip
                check("C-skip", norm, "z0 outside V in the no-norm form")
                continue
            Zp = z0                                                         # a subspace of V cap Z(B)
            Wb = span(V - Zp @ (Zp.T @ V))
            check("C-dim", Wb.shape[1] == V.shape[1] - 1, f"{Wb.shape[1]} vs {V.shape[1]}")
            epsadd = None
            if norm:
                en = [((h @ z0) ** 2).ravel() for h in hs]
                check("C-const", all(np.ptp(x) < 1e-9 for x in en))
                epsadd = [x[0] for x in en]
            same(net, cut_net(net, Wb, epsadd), X, "C-quot" + ("N" if norm else "0"))

# E: Theorem TV19.2, the lift direction
for inst in range(N_INST):
    for norm in (False, True):
        for act in ("tanh", "relu"):
            net, X, EN, new_w = make_build(norm, act, r=2, kept_frac=0.25)
            nk = int(round(q * 0.25))
            dprime = min(d, EN.shape[1] + 3 * 2)
            # Q: orthonormal columns whose range holds E_N, padded with random directions
            Q = span(np.column_stack([EN, rng.normal(size=(d, dprime - EN.shape[1]))]))[:, :dprime]
            check("E-range", np.abs(EN - Q @ (Q.T @ EN)).max() < 1e-9)
            e = net["e"]
            narrow_blocks, wide_blocks = [], []
            for (R, b, W, c) in net["blocks"]:
                nn = R.shape[0] - nk
                Rn = rng.normal(size=(nn, dprime)) / np.sqrt(dprime)
                Wn = rng.normal(size=(dprime, nn)) / np.sqrt(q)
                narrow_blocks.append((np.vstack([R[:nk] @ Q, Rn]), b, np.hstack([Q.T @ W[:, :nk], Wn]), Q.T @ c))
                wide_blocks.append((np.vstack([R[:nk], Rn @ Q.T]), b, np.hstack([W[:, :nk], Q @ Wn]), c))
            Uf = rng.normal(size=(C, dprime))
            narrow = dict(e=lambda X: e(X) @ Q, blocks=narrow_blocks, U=Uf, cy=net["cy"], eps=net["eps"], norm=norm, act=act)
            wide = dict(e=e, blocks=wide_blocks, U=Uf @ Q.T, cy=net["cy"], eps=net["eps"], norm=norm, act=act)
            same(narrow, wide, X, "E-lift")

# D: Proposition TV19.4
Xp = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
U = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
P3 = np.eye(3) - np.ones((3, 3)) / 3
D1, D2 = P3 @ U @ Xp[0], P3 @ U @ Xp[1]
check("D-indep", np.linalg.matrix_rank(np.column_stack([D1, D2])) == 2)


def kl_rows(yN, yB):
    pN = np.exp(yN - yN.max(1, keepdims=True)); pN /= pN.sum(1, keepdims=True)
    lB = yB - yB.max(1, keepdims=True); lB = lB - np.log(np.exp(lB).sum(1, keepdims=True))
    lN = np.log(pN)
    return (pN * (lN - lB)).sum(1).mean()


yN = Xp @ U.T
best = np.inf
for t in range(300):   # width 1: y = u z(x) + c, z any real per point; random restarts + gradient steps
    u, c, z = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
    for it in range(400):
        yB = z[:, None] * u[None, :] + c
        pB = np.exp(yB - yB.max(1, keepdims=True)); pB /= pB.sum(1, keepdims=True)
        pN = np.exp(yN - yN.max(1, keepdims=True)); pN /= pN.sum(1, keepdims=True)
        G = (pB - pN) / 3
        u -= 0.5 * (G * z[:, None]).sum(0); c -= 0.5 * G.sum(0); z -= 0.5 * (G @ u)
    best = min(best, kl_rows(yN, z[:, None] * u[None, :] + c))
check("D-pos", best > 1e-3, f"best={best:.3e}")

print("checks:", COUNT)
print("width-1 best D_P in the TV19.4 example:", f"{best:.4f}")
print("failures:", len(fails))
for f in fails[:20]:
    print("  ", f)
