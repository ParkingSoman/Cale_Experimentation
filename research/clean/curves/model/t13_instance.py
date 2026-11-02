# serves: T13
"""Exact instance of a spatial gated chain (COARSE-THEORY.md Definition D13, Theorem T13).

Positions: a line of L = 10 (zero padding at the ends), window Q = {-1, 0, 1}. N = 6 units per layer, depth m = 3,
n_y = 7 logits, rho uniform. The first family's pre-activations z(x, p) are given directly (a(x, p) = indicator of
(x, p), K = the table), 3000 inputs. Two gate kinds:
  path: g_l(x, p) = softplus of a fixed random function of (x, p) (a gate path that builds keep);
  self: g_1 = relu(alpha z . s_1 + beta), g_{l+1} = relu(alpha v_l . s_{l+1} + beta), one read column per gate.

Predictions (written before the run):
  S1  at every k = 1, ..., 6 and both gate kinds: the fold to the visible subspaces computes B_l v_l at every layer and
      position (relative error about 1e-13), its gates equal the network's (about 1e-13), and
      D2 = 1/2 sum_{j > k} mu_j (about 1e-12 relative).
  S2  the visible widths d_l grow toward the input: d_3 = k, and d_2, d_1 >= the width above, up to N = 6.
      With pointwise edges (Q = {0}) and the self gates: d_l <= k + (number of gate reads at layers >= l).
  S3  nesting: O_l(k) lies in O_l(k') for k < k' (residual of projecting the smaller basis on the larger, about 1e-13),
      and the stored largest build, run with prefix lengths d_l(k), gives the width-k value (about 1e-12).
Added after referee report 6 (T13 restated with reachable spaces, part 4 new), written before this run:
  S4  the T13 build of the revised statement (rows in O_l whose restrictions to Rch_l are a basis; top rows
      e_j^T P_perp H; edges solved on Rch_l): v'_l = B_l v_l and D2 = 1/2 sum_{j > k} mu_j, as in S1, in all four cases.
  S5  the referee's chain: pointwise, self-gated, widths (1, 5, 1), gate 3 reads all of v_2 (S_3 = I_5, phi of the sum),
      k = 1, r = 1: dim O_2 = 5 > k + r, but the visible width d_2 (restricted to Rch_2, of dimension 1) is 1 <= k + r,
      and the revised T13 build of widths (d_1, d_2, d_3) reaches 1/2 sum_{j > 1} mu_j.
"""
import json
import numpy as np

rng = np.random.default_rng(13)
L, N, m, ny, S = 10, 6, 3, 7, 3000
rho = np.full(L, 1.0 / L)
P = np.eye(ny) - np.ones((ny, ny)) / ny


def relu(t):
    return np.maximum(t, 0.0)


def softplus(t):
    return np.log1p(np.exp(t))


def shift(v, q):
    """v(x, p + q) with zero padding; v has shape (S, n, L)."""
    out = np.zeros_like(v)
    if q > 0:
        out[..., :-q] = v[..., q:]
    elif q < 0:
        out[..., -q:] = v[..., :q]
    else:
        out = v.copy()
    return out


z = rng.normal(size=(S, N, L)) + 0.3 * rng.normal(size=(S, 1, L)) + 0.2
gate_path = [softplus(rng.normal(size=(S, L))) for _ in range(m)]
reads = [rng.normal(size=(N, 1)) for _ in range(m)]


def run(Q, ws, Hm, c, zz, kind, Ss, alpha=1.0, beta=0.5):
    """Forward pass of a spatial gated chain (or a build) with offsets Q; returns y, gates, layers."""
    if kind == "path":
        g = gate_path[0][:, None, :]
    else:
        g = relu(alpha * np.einsum("ar,sap->sp", Ss[0], zz)[:, None, :] + beta)
    gates, v = [g], g * zz
    vs = [v]
    for l in range(m - 1):
        pre = sum(np.einsum("ab,sbp->sap", ws[l][qi], shift(v, q)) for qi, q in enumerate(Q))
        if kind == "path":
            g = gate_path[l + 1][:, None, :]
        else:
            g = relu(alpha * np.einsum("ar,sap->sp", Ss[l + 1], v)[:, None, :] + beta)
        gates.append(g)
        v = g * pre
        vs.append(v)
    y = np.einsum("ya,sap,p->sy", Hm, v, rho[:v.shape[2]]) + c
    return y, gates, vs


def rowbasis(Mrows, tol=1e-10, scale=None):
    """Orthonormal basis (as rows) of the row space of Mrows; singular values below tol * scale count as zero."""
    if Mrows.shape[0] == 0:
        return np.zeros((0, Mrows.shape[1]))
    U, s, Vt = np.linalg.svd(Mrows, full_matrices=False)
    ref = s[0] if scale is None else scale
    rk = int(np.sum(s > tol * max(ref, 1e-300)))
    return Vt[:rk]


def extend(Bsmall, Mrows, tol=1e-10):
    """Rows: Bsmall first, then an orthonormal completion so that the rows span rowspace(Bsmall) + rowspace(Mrows)."""
    R = Mrows - (Mrows @ Bsmall.T) @ Bsmall if Bsmall.shape[0] else Mrows
    add = rowbasis(R, tol, scale=np.linalg.norm(Mrows, 2)) if R.shape[0] else np.zeros((0, Mrows.shape[1]))
    return np.concatenate([Bsmall, add], axis=0)


def visible(Q, ws, Hm, Uk, Ss, kind, prev=None):
    """O_m..O_1 for U_k; prev = the bases for a smaller k (to build adapted, nested bases)."""
    m = len(ws) + 1
    dims = [ws[l][0].shape[1] for l in range(m - 1)] + [Hm.shape[1]]
    Pm = np.eye(Hm.shape[0]) - np.ones((Hm.shape[0],) * 2) / Hm.shape[0]
    B = [None] * m
    top = Uk.T @ Pm @ Hm
    B[m - 1] = extend(prev[m - 1] if prev else np.zeros((0, dims[m - 1])), top)
    for l in range(m - 2, -1, -1):
        rows = [B[l + 1] @ ws[l][qi] for qi in range(len(Q))]
        if kind == "self":
            rows.append(Ss[l + 1].T)
        if l == 0 and kind == "self":
            rows.append(Ss[0].T)
        B[l] = extend(prev[l] if prev else np.zeros((0, dims[l])), np.concatenate(rows, axis=0))
    return B


def colbasis(Acts, tol=1e-10):
    """Orthonormal basis (columns) of the span of activation vectors Acts (S, n, L)."""
    Mx = Acts.transpose(1, 0, 2).reshape(Acts.shape[1], -1)
    U, s, _ = np.linalg.svd(Mx, full_matrices=False)
    return U[:, : int(np.sum(s > tol * s[0]))]


def revised_fold(Q, ws, Hm, c, Uk, B, Ss, kind, Ey, zz, vs):
    """T13 as restated after referee report 6: rows in O_l whose restrictions to Rch_l are a basis."""
    m = len(ws) + 1
    ny_ = Hm.shape[0]
    Pm = np.eye(ny_) - np.ones((ny_, ny_)) / ny_
    Qr = [colbasis(zz)] + [colbasis(vs[l]) for l in range(1, m)]
    Bt = []
    for l in range(m):
        if l == m - 1:
            Bt.append(Uk.T @ Pm @ Hm)
            continue
        U, s, _ = np.linalg.svd(B[l] @ Qr[l], full_matrices=False)
        rk = int(np.sum(s > 1e-10 * s[0]))
        Bt.append(U[:, :rk].T @ B[l])
    wsp = [[np.linalg.lstsq((Bt[l] @ Qr[l]).T, (Bt[l + 1] @ ws[l][qi] @ Qr[l]).T, rcond=None)[0].T
            for qi in range(len(Q))] for l in range(m - 1)]
    Sp = None
    if kind == "self":
        Bfor, Qfor = [Bt[0]] + Bt[:m - 1], [Qr[0]] + Qr[:m - 1]
        Sp = [np.linalg.lstsq((Bfor[l] @ Qfor[l]).T, Qfor[l].T @ Ss[l], rcond=None)[0] for l in range(m)]
    cp = c + (np.eye(ny_) - Uk @ Uk.T @ Pm) @ (Ey - c)
    return Bt, wsp, Sp, cp, [q.shape[1] for q in Qr]


def fold(Q, ws, Hm, c, Uk, B, Ss, kind, Ey):
    Kp = B[0]                                                       # z' = B_1 z
    wsp = [[np.linalg.lstsq(B[l].T, (B[l + 1] @ ws[l][qi]).T, rcond=None)[0].T for qi in range(len(Q))]
           for l in range(m - 1)]
    Cm = np.linalg.lstsq(B[m - 1].T, (Uk.T @ P @ Hm).T, rcond=None)[0].T
    Hp = Uk @ Cm
    cp = c + (np.eye(ny) - Uk @ Uk.T @ P) @ (Ey - c)
    Sp = None
    if kind == "self":
        Bfor = [B[0]] + B[:m - 1]                                   # gate 1 reads z' (layer-1 basis), gate l+1 reads v'_l
        Sp = [np.linalg.lstsq(Bfor[l].T, Ss[l], rcond=None)[0] for l in range(m)]
    resid = max(np.abs(B[l + 1] @ ws[l][qi] - wsp[l][qi] @ B[l]).max() for l in range(m - 1) for qi in range(len(Q)))
    return Kp, wsp, Hp, cp, Sp, resid


res = {}
for Q in ([-1, 0, 1], [0]):
    for kind in ("path", "self"):
        ws = [[rng.normal(size=(N, N)) / np.sqrt(N * len(Q)) for _ in Q] for _ in range(m - 1)]
        Hm, c = rng.normal(size=(ny, N)), rng.normal(size=ny)
        y, gates, vs = run(Q, ws, Hm, c, z, kind, reads)
        yc = (y - y.mean(0)) @ P
        Cy = yc.T @ yc / S
        mu, E = np.linalg.eigh(Cy); o = np.argsort(mu)[::-1]; mu, E = mu[o], E[:, o]
        rows, prev, Bs = {}, None, {}
        for k in range(1, ny):
            Uk = E[:, :k]
            B = visible(Q, ws, Hm, Uk, reads, kind, prev)
            Bs[k] = B
            Kp, wsp, Hp, cp, Sp, resid = fold(Q, ws, Hm, c, Uk, B, reads, kind, y.mean(0))
            yb, gb, vb = run(Q, wsp, Hp, cp, np.einsum("ai,sip->sap", Kp, z), kind, Sp)
            target = 0.5 * np.sum(mu[k:])
            d2 = 0.5 * np.mean(np.sum(((yb - y) @ P) ** 2, axis=1))
            rows[k] = {"widths d_1..d_m": [b.shape[0] for b in B], "edge solve residual": float(resid),
                       "layer err": float(max(np.abs(vb[l] - np.einsum("ai,sip->sap", B[l], vs[l])).max() / np.abs(vs[l]).max()
                                              for l in range(m))),
                       "gate err": float(max(np.abs(a - b).max() / np.abs(a).max() for a, b in zip(gates, gb))),
                       "D2": float(d2), "target": float(target), "rel": float(abs(d2 - target) / max(target, 1e-300))}
            Bt, wspr, Spr, cpr, rch = revised_fold(Q, ws, Hm, c, Uk, B, reads, kind, y.mean(0), z, vs)
            ybr, gbr, vbr = run(Q, wspr, Uk, cpr, np.einsum("ai,sip->sap", Bt[0], z), kind, Spr)
            d2r = 0.5 * np.mean(np.sum(((ybr - y) @ P) ** 2, axis=1))
            rows[k]["S4 revised widths"] = [b.shape[0] for b in Bt]
            rows[k]["S4 reachable dims"] = rch
            rows[k]["S4 rel"] = float(abs(d2r - target) / max(target, 1e-300))
            rows[k]["S4 layer err"] = float(max(np.abs(vbr[l] - np.einsum("ai,sip->sap", Bt[l], vs[l])).max() / np.abs(vs[l]).max()
                                              for l in range(m)))
            if prev is not None:
                rows[k]["nesting residual"] = float(max(np.abs(prev[l] - prev[l] @ B[l].T @ B[l]).max() for l in range(m)))
            prev = B
        # S3 prefix run of the largest build
        K = ny - 1
        Bk = Bs[K]
        Kp, wsp, Hp, cp, Sp, _ = fold(Q, ws, Hm, c, E[:, :K], Bk, reads, kind, y.mean(0))
        Cbig = np.linalg.lstsq(Bk[m - 1].T, (E[:, :K].T @ P @ Hm).T, rcond=None)[0].T
        pref = {}
        for k in range(1, K + 1):
            dl = [Bs[k][l].shape[0] for l in range(m)]
            wsk = [[w[:dl[l + 1], :dl[l]] for w in wsp[l]] for l in range(m - 1)]
            Hk = E[:, :k] @ Cbig[:k, :dl[m - 1]]
            ck = c + (np.eye(ny) - E[:, :k] @ E[:, :k].T @ P) @ (y.mean(0) - c)
            Spk = None if Sp is None else [Sp[l][:([dl[0]] + dl[:m - 1])[l]] for l in range(m)]
            yb, _, _ = run(Q, wsk, Hk, ck, np.einsum("ai,sip->sap", Kp[:dl[0]], z), kind, Spk)
            d2 = 0.5 * np.mean(np.sum(((yb - y) @ P) ** 2, axis=1))
            pref[k] = float(abs(d2 - 0.5 * np.sum(mu[k:])) / max(0.5 * np.sum(mu[k:]), 1e-300))
        res[f"Q={Q} gates={kind}"] = {"mu": mu.tolist(), "per k": rows, "prefix-run rel err": pref}
# S5: the referee's chain, widths (1, 5, 1), pointwise, self-gated, gate 3 reads all of v_2
zw = rng.normal(size=(S, 1, L)) + 0.3
wsw = [[rng.normal(size=(5, 1))], [rng.normal(size=(1, 5))]]
Ssw = [np.ones((1, 1)), np.ones((1, 1)), np.eye(5)]
Hw, cw = rng.normal(size=(3, 1)), rng.normal(size=3)
Pw = np.eye(3) - np.ones((3, 3)) / 3
yw, gw, vw = run([0], wsw, Hw, cw, zw, "self", Ssw)
ycw = (yw - yw.mean(0)) @ Pw
muw, Ew = np.linalg.eigh(ycw.T @ ycw / S); o = np.argsort(muw)[::-1]; muw, Ew = muw[o], Ew[:, o]
Uw = Ew[:, :1]
Bw = visible([0], wsw, Hw, Uw, Ssw, "self")
Btw, wspw, Spw, cpw, rchw = revised_fold([0], wsw, Hw, cw, Uw, Bw, Ssw, "self", yw.mean(0), zw, vw)
ybw, _, _ = run([0], wspw, Uw, cpw, np.einsum("ai,sip->sap", Btw[0], zw), "self", Spw)
d2w = 0.5 * np.mean(np.sum(((ybw - yw) @ Pw) ** 2, axis=1))
res["S5 referee chain"] = {"dim O_l": [b.shape[0] for b in Bw], "reachable dims": rchw,
                           "visible widths d_l": [b.shape[0] for b in Btw], "k + r": 2,
                           "D2": float(d2w), "target": float(0.5 * np.sum(muw[1:])),
                           "rel": float(abs(d2w - 0.5 * np.sum(muw[1:])) / (0.5 * np.sum(muw[1:])))}
print(json.dumps(res, indent=1))
json.dump(res, open(__file__.replace(".py", ".json"), "w"), indent=1)
