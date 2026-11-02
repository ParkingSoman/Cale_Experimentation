"""Grafting (I4, recursive generalization as an axiom). Family "graft". Base model + public text only.

A coarse neuron = a trunk that has grown offshoots into its siblings' territory, each offshoot arriving by its own
route (an upstream neuron that fires where the sibling fires).
1. Trunks: the cut's top-K neurons by output energy.
2. Siblings: the next M_PER_K*K removed neurons. Each goes to the trunk that is "first in line": the largest
   mean |hidden| at the sibling's top positions, relative to the trunk's own top level.
3. Route: an upstream neuron (layers L-6..L-1) that is selective for the sibling's top positions AND readable from
   layer L's residual (corr of its read along its write direction with its true activation).
4. Graft: add alpha * read and beta * read to the trunk's gate and up rows. alpha brings the trunk's gate at the
   sibling's top positions to FRAC of the trunk's own top gate level; beta brings |up| there to FRAC of its own top
   |up| level, with the sign that makes the trunk's write point along the sibling's write.
5. Joint least-squares refit of all writes to the original layer output (as cut_ls). Same stored numbers as the cut.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import mlp_io, orig_weights, SwiGLULayer, ls_down  # noqa: E402

METHOD = {"name": "graft_v1", "family": "graft", "minutes": 3.0}
DEBUG = []
M_PER_K, N_TOP, N_SUB, BACK, FRAC, MIN_READ = 2, 50, 12000, 6, 0.5, 0.5


def silu(z):
    return z / (1.0 + np.exp(-z))


def build(model, tok, L, K, cal, log=print, frac=None):
    frac = FRAC if frac is None else frac
    t0 = time.time(); rng = np.random.default_rng(0)
    X, _ = mlp_io(model, cal[:64], L); G, U, D = orig_weights(model, L)
    sub = np.sort(rng.choice(len(X), min(N_SUB, len(X)), replace=False))
    Xs = X[sub]; w = np.array(model.model.layers[L].post_attention_layernorm.weight, dtype=np.float64)
    GP, UP = Xs @ G.T, Xs @ U.T; H = silu(GP) * UP
    en = np.linalg.norm(H, axis=0) * np.linalg.norm(D, axis=0)
    order = np.argsort(-en); trunks = order[:K]; sibs = order[K:K + M_PER_K * K]
    sgn = np.sign(np.sum(np.maximum(H, 0) ** 2, 0) - np.sum(np.minimum(H, 0) ** 2, 0)); sgn[sgn == 0] = 1

    def top(j):
        return np.argpartition(-sgn[j] * H[:, j], N_TOP)[:N_TOP]
    Ttop = {t: top(t) for t in trunks}
    lvl_h = {t: np.mean(np.abs(H[Ttop[t], t])) for t in trunks}
    lvl_g = {t: np.mean(GP[Ttop[t], t]) for t in trunks}
    lvl_u = {t: np.mean(np.abs(UP[Ttop[t], t])) for t in trunks}
    # upstream candidates: activation and read along write direction
    ups = []
    for l in range(max(0, L - BACK), L):
        Xl, _ = mlp_io(model, cal[:64], l); Gl, Ul, Dl = orig_weights(model, l)
        A = silu(Xl[sub] @ Gl.T) * (Xl[sub] @ Ul.T)
        dn = Dl / (np.linalg.norm(Dl, axis=0, keepdims=True) + 1e-12)
        R = (Xs / w) @ dn
        Az, Rz = (A - A.mean(0)) / (A.std(0) + 1e-9), (R - R.mean(0)) / (R.std(0) + 1e-9)
        rc = np.mean(Az * Rz, 0)
        ups.append((l, A, R, rc, dn))
    log(f"  graft L{L}: upstream ready [{time.time() - t0:.0f}s]")
    Gn, Un = G[trunks].copy(), U[trunks].copy(); tpos = {t: i for i, t in enumerate(trunks)}
    n_graft = 0
    for s in sibs:
        Ts = top(s)
        rel = np.array([np.mean(np.abs(H[Ts, t])) / (lvl_h[t] + 1e-12) for t in trunks])
        t = trunks[int(np.argmax(rel))]
        best = (-np.inf, None)
        for (l, A, R, rc, dn) in ups:
            for sign in (1, -1):
                a = sign * A
                sel = (a[Ts].mean(0) - a.mean(0)) / (a.std(0) + 1e-9)
                score = np.where(np.abs(rc) >= MIN_READ, sel, -np.inf)
                j = int(np.argmax(score))
                if score[j] > best[0]:
                    best = (score[j], (l, j, sign * np.sign(rc[j]), dn[:, j], R[:, j]))
        if best[1] is None:
            continue
        l, j, rs, dvec, read = best[1]
        read = rs * read; dvec = rs * dvec
        dr = read[Ts].mean() - np.median(read)
        if dr <= 0:
            continue
        i = tpos[t]
        g_now, u_now = Xs[Ts] @ Gn[i], Xs[Ts] @ Un[i]
        alpha = max(0.0, (frac * lvl_g[t] - g_now.mean()) / dr)
        wsign = np.sign(D[:, t] @ D[:, s]) * np.sign(np.mean(UP[Ttop[t], t])) or 1.0
        beta_target = frac * lvl_u[t] * wsign
        beta = (beta_target - u_now.mean()) / dr
        h_old = silu(Xs @ Gn[i]) * (Xs @ Un[i])
        Gn[i] = Gn[i] + alpha * dvec / w
        Un[i] = Un[i] + beta * dvec / w
        h_new = silu(Xs @ Gn[i]) * (Xs @ Un[i])
        Tt = Ttop[t]; rnd = rng.choice(len(Xs), 500, replace=False)
        DEBUG.append({"L": L, "t": int(t), "s": int(s), "route": [int(l), int(j)], "alpha": float(alpha), "beta": float(beta),
                      "rc": float(abs(np.corrcoef(read, ups[[u[0] for u in ups].index(l)][1][:, j])[0, 1])),
                      "core_old": float(np.mean(np.abs(h_old[Tt]))), "core_new": float(np.mean(np.abs(h_new[Tt]))),
                      "core_corr": float(np.corrcoef(h_old[Tt], h_new[Tt])[0, 1]),
                      "sib_old": float(np.mean(np.abs(h_old[Ts]))), "sib_new": float(np.mean(np.abs(h_new[Ts]))),
                      "rnd_old": float(np.mean(np.abs(h_old[rnd]))), "rnd_new": float(np.mean(np.abs(h_new[rnd]))),
                      "corr_all": float(np.corrcoef(h_old, h_new)[0, 1])})
        n_graft += 1
    Y = (silu(X @ G.T) * (X @ U.T)) @ D.T
    Dn = ls_down(X, Gn, Un, Y).astype(np.float64)
    log(f"  graft L{L}: {n_graft} grafts onto {K} trunks [{time.time() - t0:.0f}s]")
    return SwiGLULayer(Gn, Un, Dn)
