# serves: D42
"""The cleanliness list eps(N) = (eps_V, eps_G, eps_B) of Definition D42 (COARSE-THEORY.md A.5), by sampling.

A plain ReLU network is given as layers = [(W_1, b_1), ..., (W_L, b_L)] and a readout W_out (n_y x n_L). The base
measure is lambda on X, given by a batch of samples drawn uniformly from X. Every integral of Definition D41 is a
mean over these samples, so the entries are Monte Carlo estimates, not exact values.
"""
import itertools

import numpy as np


def forward(layers, x):
    """Pre-activations z_l and outputs h_l of every hidden layer, for a batch x (n x d)."""
    zs, hs, h = [], [], x
    for W, b in layers:
        z = h @ W.T + b
        h = np.maximum(z, 0.0)
        zs.append(z)
        hs.append(h)
    return zs, hs


def readers(layers, W_out, l):
    """Reader weight rows of layer l (0-based): the next layer's W, or the logit differences for the last layer."""
    if l + 1 < len(layers):
        return layers[l + 1][0]
    n_y = W_out.shape[0]
    return np.array([W_out[c] - W_out[c2] for c, c2 in itertools.combinations(range(n_y), 2)])


def reduced(layers, W_out):
    """Reduced form of Definition D20 for a non-degenerate network: rescale every unit to |(w, b)| = 1 and move the
    positive scale into the outgoing weights. Unread units are deleted. Merging units whose rows are positive multiples is not
    implemented: such a network raises an error."""
    layers = [(W.copy(), b.copy()) for W, b in layers]
    W_out = W_out.copy()
    for l in range(len(layers) - 1, -1, -1):  # delete units that are not read (their function does not change)
        col = W_out if l == len(layers) - 1 else layers[l + 1][0]
        keep = np.linalg.norm(col, axis=0) > 0
        if keep.all():
            continue
        W, b = layers[l]
        layers[l] = (W[keep], b[keep])
        if l == len(layers) - 1:
            W_out = W_out[:, keep]
        else:
            Wn, bn = layers[l + 1]
            layers[l + 1] = (Wn[:, keep], bn)
    out, scale = [], None
    for l, (W, b) in enumerate(layers):
        rows = np.hstack([W, b[:, None]])
        for u, v in itertools.combinations(range(rows.shape[0]), 2):
            ru, rv = rows[u] / np.linalg.norm(rows[u]), rows[v] / np.linalg.norm(rows[v])
            if np.allclose(ru, rv, atol=1e-12):  # positive multiples: reduced form merges them
                raise ValueError(f"layer {l}: units {u}, {v} are positive multiples; reduced form would merge them")
    for W, b in layers:
        W = W.copy() if scale is None else W * scale[None, :]
        n = np.linalg.norm(np.hstack([W, b[:, None]]), axis=1)
        out.append((W / n[:, None], b / n))
        scale = n
    return out, W_out * scale[None, :]


def eps_v_reader(layers, W_out, x, tol=1e-12, reduce=True):
    """Candidate form of V (comment (h) of D42): each live reader's cancelled share, averaged over the reader's OWN
    target, weighted by the reader's use. For a hidden reader the target has density h_r; a logit difference
    y_c - y_c' is a reader in both orientations, with target density (y_c - y_c')_+."""
    if reduce:
        layers, W_out = reduced(layers, W_out)
    zs, hs = forward(layers, x)
    L = len(layers)
    rows = []  # (use, mean cancelled share on own target)
    for l in range(L):
        h = hs[l]
        if l + 1 < L:
            R, act = layers[l + 1][0], hs[l + 1]
            col = layers[l + 2][0] if l + 2 < L else W_out
            use = [act[:, r].mean() * np.linalg.norm(col[:, r]) for r in range(R.shape[0])]
        else:
            n_y = W_out.shape[0]
            pairs = [(c, c2) for c in range(n_y) for c2 in range(n_y) if c != c2]
            R = np.array([W_out[c] - W_out[c2] for c, c2 in pairs])
            act = np.maximum(h @ R.T, 0.0)
            use = [act[:, k].mean() for k in range(R.shape[0])]
        A = h[:, None, :] * R[None, :, :]
        S = np.abs(A).sum(axis=2)
        kap = np.where(S > tol, (S - np.abs(A.sum(axis=2))) / np.where(S > tol, S, 1.0), 0.0)
        for k in range(R.shape[0]):
            if act[:, k].sum() <= tol:
                continue
            w = act[:, k] / act[:, k].sum()
            rows.append((use[k], float(w @ kap[:, k])))
    tot = sum(r[0] for r in rows)
    return 0.0 if tot <= tol else sum(r[0] * r[1] for r in rows) / tot


def b_last(layers, W_out, x, tol=1e-12, reduce=True):
    """Candidate extension of B to the last hidden layer (comment (h) of D42): b_u = the least, over the logit
    differences y_c - y_c' (both orientations) that read u with a positive weight, of mu_u({(y_c - y_c')(x) <= 0}).
    Returns the use-weighted mean over live last-layer units."""
    if reduce:
        layers, W_out = reduced(layers, W_out)
    zs, hs = forward(layers, x)
    h = hs[-1]
    n_y = W_out.shape[0]
    pairs = [(c, c2) for c in range(n_y) for c2 in range(n_y) if c != c2]
    R = np.array([W_out[c] - W_out[c2] for c, c2 in pairs])
    gap = h @ R.T
    P = np.eye(n_y) - 1.0 / n_y
    col = P @ W_out
    num = den = 0.0
    for u in range(h.shape[1]):
        M = h[:, u].mean()
        if M <= tol:
            continue
        w = h[:, u] / h[:, u].sum()
        pos = np.where(R[:, u] > tol)[0]
        bu = 1.0 if len(pos) == 0 else min(float(w @ (gap[:, k] <= tol)) for k in pos)
        om = M * np.linalg.norm(col[:, u])
        num += om * bu
        den += om
    return 0.0 if den <= tol else num / den


def eps(layers, W_out, x, tol=1e-12, reduce=True):
    """Return dict with eps_V, eps_G, eps_B and the per-unit pieces, from samples x of the base measure.
    With reduce=True (Definition D41) the entries are computed on the reduced form."""
    if reduce:
        layers, W_out = reduced(layers, W_out)
    zs, hs = forward(layers, x)
    L = len(layers)
    P = np.eye(W_out.shape[0]) - 1.0 / W_out.shape[0]
    units = []  # (l, u, omega, v_term, g, b)
    for l in range(L):
        h = hs[l]
        R = readers(layers, W_out, l)
        # cancelled share of every reader at every sample
        A = h[:, None, :] * R[None, :, :]  # n x readers x units
        S = np.abs(A).sum(axis=2)
        kap = np.where(S > tol, (S - np.abs(A.sum(axis=2))) / np.where(S > tol, S, 1.0), 0.0)
        col = (P @ W_out) if l == L - 1 else layers[l + 1][0]
        live_r = np.ones(R.shape[0], bool) if l == L - 1 else (hs[l + 1].sum(axis=0) > tol)
        for u in range(h.shape[1]):
            M = h[:, u].mean()
            if M <= tol:
                continue
            w = h[:, u] / h[:, u].sum()  # target mu_u on the samples
            omega = M * np.linalg.norm(col[:, u])
            ra = np.abs(R[:, u]) * live_r  # only live readers are counted (Definition D41)
            v_term = 0.0 if ra.sum() <= tol else float((ra / ra.sum()) @ (kap.T @ w))
            g = None
            if l >= 1:
                pos = np.where(layers[l][0][u] > tol)[0]
                g = 1.0 if len(pos) == 0 else min(float(w @ (hs[l - 1][:, v] <= tol)) for v in pos)
            bb = None
            if l <= L - 2:
                pos = np.where(layers[l + 1][0][:, u] > tol)[0]
                bb = 1.0 if len(pos) == 0 else min(float(w @ (hs[l + 1][:, r] <= tol)) for r in pos)
            units.append((l, u, omega, v_term, g, bb))

    def wmean(rows, k):
        rows = [r for r in rows if r[k] is not None]
        tot = sum(r[2] for r in rows)
        return 0.0 if tot <= tol else sum(r[2] * r[k] for r in rows) / tot

    return dict(eps_V=wmean(units, 3), eps_G=wmean(units, 4), eps_B=wmean(units, 5), units=units)


def sample_box(lo, hi, n, rng):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return lo + (hi - lo) * rng.random((n, lo.size))


def sample_ball(center, radius, n, rng):
    d = len(center)
    g = rng.standard_normal((n, d))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    r = radius * rng.random(n) ** (1.0 / d)
    return np.asarray(center, float) + g * r[:, None]
