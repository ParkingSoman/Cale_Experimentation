# serves: TQ31
"""Small plain ReLU networks and the principle errors of D42 and D45, computed on a finite base law.

A net is (layers, Wout, c). layers[l] = (W, b): pre-activation of layer l is W h_(l-1) + b, h_(-1) = input.
Canonical form of D43 is applied before anything is computed.
Base law nu = points X with weights p. Everything is exact on this law (no sampling).
"""
import itertools

import numpy as np


def softmax(y):
    y = y - y.max(-1, keepdims=True)
    e = np.exp(y)
    return e / e.sum(-1, keepdims=True)


def forward(layers, Wout, c, X):
    hs, zs = [], []
    h = X
    for W, b in layers:
        z = h @ W.T + b
        h = np.maximum(z, 0)
        zs.append(z)
        hs.append(h)
    return hs, zs, h @ Wout.T + c


def canonical(layers, Wout, c):
    """D43 item 1: rescale, merge, delete, until nothing changes."""
    layers = [(W.copy(), b.copy()) for W, b in layers]
    Wout = Wout.copy()
    while True:
        changed = False
        for l in range(len(layers)):
            W, b = layers[l]
            nxt = layers[l + 1][0] if l + 1 < len(layers) else Wout
            # delete units that are not read
            keep = np.abs(nxt).sum(0) > 1e-13
            if not keep.all():
                changed = True
            W, b, nxt = W[keep], b[keep], nxt[:, keep]
            # rescale
            nrm = np.sqrt((W ** 2).sum(1) + b ** 2)
            nz = nrm > 0
            W[nz] = W[nz] / nrm[nz, None]
            b[nz] = b[nz] / nrm[nz]
            nxt = nxt.copy()
            nxt[:, nz] = nxt[:, nz] * nrm[nz]
            if np.abs(nrm[nz] - 1).max(initial=0) > 1e-12:
                changed = True
            # merge
            sig = np.round(np.c_[W, b], 10)
            uniq, inv = np.unique(sig, axis=0, return_inverse=True)
            inv = np.ravel(inv)
            if len(uniq) < len(sig):
                changed = True
                first = [np.where(inv == i)[0][0] for i in range(len(uniq))]
                W2, b2 = W[first], b[first]
                nx2 = np.zeros((nxt.shape[0], len(uniq)))
                np.add.at(nx2.T, inv, nxt.T)
                W, b, nxt = W2, b2, nx2
            layers[l] = (W, b)
            if l + 1 < len(layers):
                layers[l + 1] = (nxt, layers[l + 1][1])
            else:
                Wout = nxt
        if not changed:
            return layers, Wout, c


def kappa(votes):
    """votes: (points, readers, units). cancelled share of D41, per point and reader."""
    s = np.abs(votes).sum(-1)
    t = np.abs(votes.sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        k = (s - t) / s
    k[s <= 1e-14] = 0
    return np.clip(k, 0, 1)


def principles(layers, Wout, c, X, p, want_xi=True, tol=1e-10):
    """Returns the lists of D42 and D45 on base law (X, p), plus per-unit tables."""
    layers, Wout, c = canonical(layers, Wout, c)
    hs, zs, y = forward(layers, Wout, c, X)
    q = softmax(y)
    L = len(layers)
    C = Wout.shape[0]
    Wt = Wout - Wout.mean(0, keepdims=True)  # P_perp W_out
    pairs = np.array(list(itertools.combinations(range(C), 2)))
    out = {"units": [h.shape[1] for h in hs]}
    V_num = V_den = G_num = G_den = B_num = B_den = 0.0
    Vrel_num = Vrel_den = 0.0
    pack_num = pack_den = unread = xi_bad = om_tot = 0.0
    per = []
    for l in range(L):
        h = hs[l]
        mass = h.T @ p
        live = mass > 1e-14
        last = l == L - 1
        if last:
            Wr = Wout[pairs[:, 0]] - Wout[pairs[:, 1]]
            votes = Wr[None, :, :] * h[:, None, :]
            kap = kappa(votes)
            relw = q[:, pairs[:, 0]] * q[:, pairs[:, 1]]
        else:
            Wnext = layers[l + 1][0]
            lv = (hs[l + 1].T @ p) > 1e-14
            Wr = Wnext[lv]
            votes = Wr[None, :, :] * h[:, None, :]
            kap = kappa(votes)
            relw = None
        # builders' votes for routes
        prev = X if l == 0 else hs[l - 1]
        Wb = layers[l][0]
        for u in np.where(live)[0]:
            mu = p * h[:, u] / mass[u]
            col = Wt[:, u] if last else layers[l + 1][0][:, u]
            om_V = mass[u] * np.linalg.norm(col)
            r_abs = np.abs(Wr[:, u])
            pi_ru = r_abs / r_abs.sum() if r_abs.sum() > 0 else r_abs * 0
            Vu = float(pi_ru @ (kap.T @ mu))
            V_num += om_V * Vu
            V_den += om_V
            if last:
                # relevance-weighted V (mine, not in D42): pairs weighted by q_c q_c' at x, and by how much u moves the pair
                w = relw * r_abs[None, :]
                den = w.sum(1)
                k_rel = (w * kap).sum(1) / np.where(den > 0, den, 1)
                Vrel_num += om_V * float(mu @ k_rel)
                Vrel_den += om_V
            # G
            if l >= 1:
                Wu = layers[l][0][u]
                vs = np.where(Wu > 1e-12)[0]
                g_u = min([float(mu @ (hs[l - 1][:, v] <= 0)) for v in vs], default=1.0)
                G_num += om_V * g_u
                G_den += om_V
            else:
                g_u = None
            # B
            if not last:
                rs = np.where(layers[l + 1][0][:, u] > 1e-12)[0]
                b_u = min([float(mu @ (hs[l + 1][:, r] <= 0)) for r in rs], default=1.0)
                B_num += om_V * b_u
                B_den += om_V
            else:
                b_u = None
            # D44 use, on all points
            if last:
                g = q * (Wout[:, u][None, :] - (q @ Wout[:, u])[:, None])
                S = q @ Wt[:, u]
                T = np.abs(Wt[:, u]).sum()
            else:
                Wn = layers[l + 1][0]
                g = Wn[:, u][None, :] * (zs[l + 1] > 0)
                S = g.sum(1)
                T = np.abs(Wn[:, u]).sum()
            # routes on the target
            on = h[:, u] > 0
            vote = prev * Wb[u][None, :]
            top = vote.max(1, keepdims=True)
            routes = {}
            for i in np.where(on)[0]:
                key = tuple(np.where(vote[i] >= top[i] - tol)[0])
                routes.setdefault(key, []).append(i)
            sum_m = np.zeros(g.shape[1])
            sum_abs = 0.0
            for key, idx in routes.items():
                wg = mu[idx].sum()
                mg = (mu[idx][:, None] * g[idx]).sum(0) / wg
                sum_m += wg * mg
                sum_abs += wg * np.linalg.norm(mg)
            pi_u = 1 - (sum_m @ sum_m) / sum_abs ** 2 if sum_abs > 1e-14 else 0.0
            lam = float((mu - p) @ S) / T if T > 0 else 0.0
            om45 = float(p @ (h[:, u] * np.linalg.norm(g, axis=1)))
            pack_num += om45 * pi_u
            pack_den += om45
            unread += om45 * (lam <= 1e-13)
            om_tot += om45
            xi = None
            if want_xi and T > 0:
                if last:
                    R = y - Wout[:, u][None, :] * h[:, u][:, None]
                    # S(t, R) on all pairs of base points
                    t = h[:, u]
                    A = np.zeros((len(t), len(t)))
                    for i, ti in enumerate(t):
                        A[i] = softmax(Wout[:, u][None, :] * ti + R) @ Wt[:, u]
                else:
                    Wn = layers[l + 1][0]
                    R = zs[l + 1] - Wn[:, u][None, :] * h[:, u][:, None]
                    t = h[:, u]
                    A = np.zeros((len(t), len(t)))
                    for i, ti in enumerate(t):
                        A[i] = ((Wn[:, u][None, :] * ti + R > 0) * Wn[:, u][None, :]).sum(1)
                lam_ind = float((mu - p) @ A @ p) / T
                xi = lam - lam_ind
                xi_bad += om45 * (xi <= 1e-13)
            per.append(dict(layer=l, unit=int(u), mass=float(mass[u]), V=Vu, g=g_u, b=b_u, pi=float(pi_u), lam=lam, xi=xi, routes=len(routes)))
    out["eps_V"] = V_num / V_den if V_den else 0.0
    out["eps_V_rel"] = Vrel_num / Vrel_den if Vrel_den else 0.0
    out["eps_G"] = G_num / G_den if G_den else None
    out["eps_B"] = B_num / B_den if B_den else None
    out["eps_pack"] = pack_num / pack_den if pack_den else 0.0
    out["eps_unread"] = unread / om_tot if om_tot else 0.0
    out["eps_agree"] = xi_bad / om_tot if om_tot else 0.0
    out["per"] = per
    return out


def least_uncut(n, edges):
    best, arg = 10 ** 9, None
    for s in itertools.product((1, -1), repeat=n - 1):
        s = (1,) + s
        v = sum(1 for a, b in edges if s[a] == s[b])
        if v < best:
            best, arg = v, s
    return best, np.array(arg, dtype=float)


def subdivide(n, edges):
    """Replace each edge by a path of 3 edges, as in TQ31 part 3. U is unchanged."""
    out = []
    k = n
    for a, b in edges:
        out += [(a, k), (k, k + 1), (k + 1, b)]
        k += 2
    return k, out


def cycle(n):
    return n, [(i, (i + 1) % n) for i in range(n)]


def complete(n):
    return n, list(itertools.combinations(range(n), 2))


def tq31(n, edges, a, rho, D):
    """Table of Proposition TQ31 in the form of model/tq31_anytail.py: classes g, b_1..b_n, block, filler."""
    deg = np.zeros(n)
    M = np.zeros((n, n))
    W = np.ones((n, n))
    for u, v in edges:
        M[u, v] = M[v, u] = -1
        W[u, v] = W[v, u] = D
        deg[u] += 1
        deg[v] += 1
    for v in range(n):
        M[v, v] = 1
        W[v, v] = D * max(deg[v], 1)
    rs = W.sum(1)
    S = rs.max() + n
    Fill = np.ones((n, n)) + np.diag(rs.max() - rs)
    Wa = np.c_[W, Fill]
    Ma = np.c_[M, np.zeros((n, n))]
    B = float(np.log(S * (1 - rho) / rho / (1 + 1 / np.e)))
    C = 1 + n + 2 * n
    y0 = np.zeros((n, C))
    y0[:, 0] = B - 1
    y0[np.arange(n), 1 + np.arange(n)] = B
    y0[:, 1 + n:] = np.log(Wa)
    mv = np.zeros_like(y0)
    mv[:, 1 + n:] = Ma * a / 2
    return y0, mv


def points(m):
    """Inputs of DQ18: coordinates h_1..h_m, s_1..s_m. Order: x_1^-, x_1^+, x_2^-, ..."""
    X = np.zeros((2 * m, 2 * m))
    for w in range(m):
        X[2 * w, [w, m + w]] = 1, 1
        X[2 * w + 1, [w, m + w]] = 1, 3
    return X


def table(y0, mv):
    m = y0.shape[0]
    Y = np.zeros((2 * m, y0.shape[1]))
    Y[0::2] = y0 - mv
    Y[1::2] = y0 + mv
    return Y


def net_dq18(y0, mv):
    """N itself: h_w carries y0 - 2 mv, s_w carries mv (mv = (a/2) M)."""
    m, C = y0.shape
    W1 = np.eye(2 * m)
    b1 = np.zeros(2 * m)
    Wout = np.zeros((C, 2 * m))
    Wout[:, :m] = (y0 - 2 * mv).T
    Wout[:, m:] = mv.T
    return [(W1, b1)], Wout, np.zeros(C)
