# serves: TC1
"""Exact small check of Theorem TC1, Comment TC1.1 and Proposition TC2. It can only refute.

Network: plain ReLU layers, h_(l+1) = relu(A_l h_l + b_l), logits = W h_L + c. Finite list P with weights. N_K sets every unit outside K
to its P-mean, layer by layer. eps(c, Q): weighted square difference between z_c in N and z_c with all of the layer below outside Q set
to its P-mean (the layer below itself held at N's values).
PREDICTION, written before the first run: 0 fails.
"""
import numpy as np

fails = 0


def check(name, ok, detail=""):
    global fails
    if not ok:
        fails += 1
        print("FAIL", name, detail)


def relu(x):
    return np.maximum(x, 0)


def forward(As, bs, x, keep=None, means=None):
    """Returns pre-activations and values per layer. keep[l] = kept units of layer l. Others take means[l]."""
    h, zs, hs = x, [], []
    for l, (A, b) in enumerate(zip(As, bs)):
        z = h @ A.T + b
        h = relu(z)
        if keep is not None:
            h = h.copy()
            out = [u for u in range(h.shape[1]) if u not in keep[l]]
            h[:, out] = means[l][out]
        zs.append(z)
        hs.append(h)
    return zs, hs


def softmax(y):
    y = y - y.max(1, keepdims=True)
    e = np.exp(y)
    return e / e.sum(1, keepdims=True)


def kl(p, y0, y1):
    s, q = softmax(y0), softmax(y1)
    return float((p * (s * (np.log(s + 1e-300) - np.log(q + 1e-300))).sum(1)).sum())


def eps(As, bs, hs, means, p, l, c, Q):
    """Claim (c, Q): c in layer l + 1, Q in layer l (0-indexed lists of hidden layers)."""
    h = hs[l]
    h2 = h.copy()
    out = [u for u in range(h.shape[1]) if u not in Q]
    h2[:, out] = means[l][out]
    z, z2 = h @ As[l + 1].T[:, :] if False else h @ As[l + 1].T + bs[l + 1], h2 @ As[l + 1].T + bs[l + 1]
    return float((p * (z[:, c] - z2[:, c]) ** 2).sum())


rng = np.random.default_rng(0)
for seed in range(200):
    rng = np.random.default_rng(seed)
    n, d, w = 30, 5, [6, 6, 6]
    x = rng.normal(size=(n, d))
    p = rng.random(n) + .05
    p /= p.sum()
    As, bs, keep = [], [], []
    dims = [d] + w
    for l in range(3):
        keep.append(sorted(rng.choice(w[l], size=int(rng.integers(2, 5)), replace=False).tolist()))
    for l in range(3):
        A = rng.normal(size=(dims[l + 1], dims[l]))
        if l > 0:  # a kept unit reads only kept units of the layer below: its claim on K_l holds at 0
            for c in keep[l]:
                for u in range(dims[l]):
                    if u not in keep[l - 1]:
                        A[c, u] = 0
        As.append(A)
        bs.append(rng.normal(size=dims[l + 1]) * .3)
    W = rng.normal(size=(4, w[-1]))
    t = rng.normal(size=4)
    zs, hs = forward(As, bs, x)
    means = [(p[:, None] * h).sum(0) for h in hs]
    for l in range(2):
        for c in keep[l + 1]:
            check("claim on K at 0", eps(As, bs, hs, means, p, l, c, keep[l]) < 1e-20, (seed, l, c))
    zk, hk = forward(As, bs, x, keep, means)
    for l in range(3):
        for c in keep[l]:
            check("TC1.1 kept units unchanged", np.abs(zk[l][:, c] - zs[l][:, c]).max() < 1e-12, (seed, l, c))
    y, yk = hs[-1] @ W.T + t, hk[-1] @ W.T + t
    D = [u for u in range(w[-1]) if u not in keep[-1]]
    h_top = hs[-1].copy()
    h_top[:, D] = means[-1][D]
    check("TC1.2 whole cut = top drop", abs(kl(p, y, yk) - kl(p, y, h_top @ W.T + t)) < 1e-12, seed)
    osc = (yk - y).max(1) - (yk - y).min(1)
    check("TC1.3 Hoeffding bound", kl(p, y, yk) <= float((p * osc ** 2).sum()) / 8 + 1e-12, seed)
    # nested cuts: a smaller closed set is also exact, and removing a leaf changes nothing below it
    if len(keep[-1]) > 1:
        k2 = [keep[0], keep[1], keep[-1][:-1]]
        zk2, _ = forward(As, bs, x, k2, means)
        check("TC1.4 nesting", all(np.abs(zk2[l][:, c] - zs[l][:, c]).max() < 1e-12 for l in range(3) for c in k2[l]), seed)

# Comment TC1.1: a claim on a smaller Q does not cover a larger kept set. z = q + |u - v|, on points with u = v in {0, 2}.
q = np.array([1., 2., 3., 4.])
u = np.array([0., 2., 0., 2.])
v = u.copy()
pp = np.ones(4) / 4
z = q + np.abs(u - v)
zQ = q + np.abs(u.mean() - v.mean())
zK = q + np.abs(u - v.mean())
check("TC1.1 example: claim on Q holds", np.abs(z - zQ).max() < 1e-15)
check("TC1.1 example: larger kept set moves", np.abs(z - zK).max() > .5)

# Proposition TC2: the graded form, with claims above 0
for seed in range(200):
    rng = np.random.default_rng(1000 + seed)
    n, d, w = 30, 5, [6, 6, 6]
    x = rng.normal(size=(n, d))
    p = rng.random(n) + .05
    p /= p.sum()
    dims = [d] + w
    As = [rng.normal(size=(dims[l + 1], dims[l])) * .6 for l in range(3)]
    bs = [rng.normal(size=dims[l + 1]) * .3 for l in range(3)]
    keep = [sorted(rng.choice(w[l], size=4, replace=False).tolist()) for l in range(3)]
    W = rng.normal(size=(4, w[-1]))
    zs, hs = forward(As, bs, x)
    means = [(p[:, None] * h).sum(0) for h in hs]
    zk, hk = forward(As, bs, x, keep, means)
    # delta_l: root of the weighted mean square change of the kept units of layer l
    delta = [np.sqrt((p * ((hk[l] - hs[l])[:, keep[l]] ** 2).sum(1)).sum()) for l in range(3)]
    check("TC2 first layer unchanged", delta[0] < 1e-12, seed)
    bound = 0.0
    for l in range(2):
        e = sum(eps(As, bs, hs, means, p, l, c, keep[l]) for c in keep[l + 1])
        A_kk = As[l + 1][np.ix_(keep[l + 1], keep[l])]
        bound = np.linalg.norm(A_kk, 2) * bound + np.sqrt(e)
        check("TC2 recursion", delta[l + 1] <= bound + 1e-12, (seed, l, delta[l + 1], bound))
        # The form of the proposition after REFEREE-BCT: the raw e, also on a channel with V_c = 0 (the referee's example is below).

# Referee's corner: z_c = u - v with u = v on P has V_c = 0 and e > 0, so a bound using V_c * eps misses it, and one using e does not.
u = np.array([0., 2.])
pz = np.ones(2) / 2
z_full = u - u
z_fill = u - u.mean()
e_val = float((pz * (z_full - z_fill) ** 2).sum())
check("TC2 corner: V_c = 0", float(z_full.var()) == 0)
check("TC2 corner: e > 0", e_val > .5)

# Proposition TC3: all errors tend to 0, the loss does not move.
losses = []
for d in (1e-1, 1e-2, 1e-3, 1e-4):
    q = np.array([10., 10., 12., 12.])
    w = np.array([0., 2., 0., 2.])
    pz = np.ones(4) / 4
    c = q + d * (w - 1)
    cp = q.copy()
    ze = (c - cp) / d + 5
    check("TC3 all pass their ReLU", min(c.min(), cp.min(), ze.min()) > 0)
    ck = q + d * (w.mean() - 1)  # w filled with its mean, which is 1
    zek = (ck - cp) / d + 5
    e_c = float((pz * (c - ck) ** 2).sum())
    V_c = float((pz * c ** 2).sum() - (pz * c).sum() ** 2)
    check("TC3 eps_par(c, {q}) <= d^2", e_c / V_c <= d ** 2 + 1e-15, (d, e_c / V_c))
    check("TC3 values in N", np.abs(ze - (w + 4)).max() < 1e-8)
    check("TC3 value in N_K", np.abs(zek - 5).max() < 1e-8)
    y = np.stack([ze, 0 * ze], 1)
    yk = np.stack([zek, 0 * zek], 1)
    losses.append(kl(pz, y, yk))
check("TC3 the loss stays and is above 0", min(losses) > 1e-3 and max(losses) - min(losses) < 1e-6, losses)

# Lemma TC4: one covariance gives e for every Q, on an affine read.
for seed in range(100):
    rng = np.random.default_rng(3000 + seed)
    h = relu(rng.normal(size=(40, 7)))
    pz = rng.random(40) + .05
    pz /= pz.sum()
    A = rng.normal(size=7)
    mean = (pz[:, None] * h).sum(0)
    Cov = ((h - mean).T * pz) @ (h - mean)
    Q = rng.choice(7, size=int(rng.integers(0, 7)), replace=False).tolist()
    D = [u_ for u_ in range(7) if u_ not in Q]
    h2 = h.copy()
    h2[:, D] = mean[D]
    e_direct = float((pz * ((h - h2) @ A) ** 2).sum())
    e_cov = float(A[D] @ Cov[np.ix_(D, D)] @ A[D]) if D else 0.0
    check("TC4", abs(e_direct - e_cov) < 1e-10, (seed, e_direct, e_cov))
print("fails", fails)
