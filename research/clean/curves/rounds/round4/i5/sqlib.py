# serves: TQ31
"""The square family table: every class owns an area, no shared class g, no filler, a family part and side links in each row.

q families of size t on a graph G with q vertices, n = q t classes. Row a in family v:
   own a:        W = S - (rest of the row), M = +1
   sibling:      W = Df, M = +1
   side (b in a family linked to v by an edge of G): W = D, M = -1
   other:        W = 1, M = 0 (or M = sig_a sig_b if hole_sign is given)
Logits y0 = log W, y(x^+-) = y0 +- (a/2) M. Own mass = 1 - rho.
"""
import numpy as np


def sqfam(q, edges, t, a, rho, D, Df=None, hole_sign=None):
    n = q * t
    Df = D * np.exp(a + 1) if Df is None else Df
    fam = np.repeat(np.arange(q), t)
    adj = np.zeros((q, q), bool)
    for u, v in edges:
        adj[u, v] = adj[v, u] = True
    W = np.ones((n, n))
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if fam[i] == fam[j]:
                W[i, j], M[i, j] = Df, 1
            elif adj[fam[i], fam[j]]:
                W[i, j], M[i, j] = D, -1
            elif hole_sign is not None:
                M[i, j] = hole_sign[fam[i]] * hole_sign[fam[j]]
    np.fill_diagonal(W, 0)
    rest = W.sum(1)
    S = rest.max() / rho
    deg = adj.sum(1)[fam]
    own = S - rest
    assert (own >= D * t * deg - 1e-9).all(), "own weight too small for the count: lower rho"
    W[np.arange(n), np.arange(n)] = own
    M[np.arange(n), np.arange(n)] = 1
    return M, W, fam, adj, Df


def bounds(M, W, fam, a, D, t, U):
    """The sandwich of the square proposition: lower and upper constants, computed from the table."""
    n = len(W)
    y0 = np.log(W)
    Y = np.stack([y0 - a / 2 * M, y0 + a / 2 * M])
    e = np.exp(Y - Y.max(-1, keepdims=True))
    s = e / e.sum(-1, keepdims=True)
    taup = (s.min(0) / W).min()
    taupp = (s.max(0) / W).max()
    psi = np.exp(-a / 2) - 1 + a / 2
    lower = taup * psi * D * t ** 2 * U / (2 * n)
    gc = 2 * (np.exp(2 * a) - 1 - 2 * a) / (2 * a) ** 2
    holes = (W == 1).sum()
    upper = gc * taupp * (a ** 2 / (8 * n)) * (8 * D * t ** 2 * U + holes)
    return lower, upper
