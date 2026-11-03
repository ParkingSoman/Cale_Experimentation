# serves: TQ31
"""Step 4. Same numbers on hard tables and on ResNet-18's rows (cache/map, 20 photos per class, class rows and photo rows).

Numbers, fixed before the run. T = a row's table off its own class, normalised. K rows.
 own: mean own-class mass; std across rows. wins: share of classes that win a row. univ: the largest over classes of the 5th percentile of its mass over other rows.
 head: top 1, 5, 10 share of T; effective entries exp(entropy).
 recip: share of row top-5 pairs that are also top-5 in the reverse row. rcorr: corr of log T[c,i] and log T[i,c] on top-5 entries.
 sink: largest column in-mass over the mean.
 eps_prod: weighted rms of log T after row and column scale, by alternating weighted means (DQ16 with H_c = {own}).
 closure: rho = mean over rows of sum_i T_A[c,i] BC(T_A[c], T_B[i]), BC = sum_j sqrt(p_j q_j). self = mean BC(T_A[c], T_B[c]). Halves A, B for ResNet-18 (T_A = T_B for a noise-free table).
   eps_fam = 1 - rho / self. Null: rows of T_B shuffled. Split by link kind on ResNet-18: family link cosW >= .3, side link cosW < .1.
 spec: Sigma s^4 / Sigma s^2 of Wtilde (DQ22's matrix), cross-half on ResNet-18, using quarters: tr(Q1 Q2' Q3 Q4') / tr(Q1 Q2'). eps_close = 1 - that.
 eta: fraction of link weight uncut, on A = sym(T truncated to top 8). lower side (1 + lambda_min) / 2, upper side by local search. Also on families contracted (ResNet-18: Ward clusters of unit W, 120 families).
 capture(r): 1 - p*_r / E_W, where rows are photos (or points), M is the movement around the row's class mean, W = class row table, one free constant per photo, r free codes per photo, by weighted ALS. held = same on the half of photos V did not see, codes refit.
 movement: first principal direction of each class's movement in its W metric. Mean of M[i] / M[own] over family links and side links; share of the row's movement energy in it.
 Then eps_V, V_rel, lambda > 0 share, xi <= 0 share at ResNet-18's last hidden layer on photo rows, by sampling.
PREDICTIONS: univ far below the hard tables' .18; recip below 1. eps_fam: hard tables near 1 on side links; ResNet-18 clearly lower on family links and near hard tables on side links. eta: hard tables small, ResNet-18 above .2. capture(1): hard tables above .5, ResNet-18 below .2.
"""
import itertools
import json
import sys

import numpy as np

HERE = "/Users/shonusengupta/Cale_Experimentation/research/clean/curves"
sys.path.insert(0, f"{HERE}/rounds/round4/i5")
from netlib import cycle, complete, subdivide, softmax, tq31
from sqlib import sqfam

np.set_printoptions(precision=3, suppress=True, linewidth=160)
rng = np.random.default_rng(17)


def offtable(S):
    T = S.copy()
    np.fill_diagonal(T, 0)
    return T / T.sum(1, keepdims=True)


def basic(S, Sb=None):
    """S: K x K class rows. Sb: second half, or None."""
    K = len(S)
    T = offtable(S)
    Tb = offtable(Sb) if Sb is not None else T
    d = {}
    d["own"] = float(np.diag(S).mean())
    d["own_std"] = float(np.diag(S).std())
    off = S.copy()
    np.fill_diagonal(off, np.nan)
    d["univ"] = float(np.nanpercentile(off, 5, axis=0).max())
    srt = -np.sort(-T, 1)
    d["top1"], d["top5"], d["top10"] = float(srt[:, 0].mean()), float(srt[:, :5].sum(1).mean()), float(srt[:, :10].sum(1).mean())
    d["eff"] = float(np.exp(-(T * np.log(T + 1e-300)).sum(1)).mean())
    top = np.argsort(-T, 1)[:, :5]
    intop = np.zeros_like(T, bool)
    intop[np.arange(K)[:, None], top] = True
    d["recip"] = float((intop & intop.T).sum() / intop.sum())
    a, b = np.log(T[intop] + 1e-12), np.log(T.T[intop] + 1e-12)
    d["rcorr"] = float(np.corrcoef(a, b)[0, 1])
    d["sink"] = float((T.sum(0) / T.sum(0).mean()).max())
    # eps_prod
    w = T / K
    L = np.log(T + 1e-300)
    al, be = np.zeros(K), np.zeros(K)
    for _ in range(200):
        al = ((L - be[None, :]) * w).sum(1) / w.sum(1)
        be = ((L - al[:, None]) * w).sum(0) / np.maximum(w.sum(0), 1e-300)
    r = L - al[:, None] - be[None, :]
    d["prod_rms"] = float(np.sqrt((w * r ** 2).sum()))
    d["level_rms"] = float(np.sqrt((w * (L - (w * L).sum()) ** 2).sum()))
    # closure
    RA, RB = np.sqrt(T), np.sqrt(Tb)
    BC = RA @ RB.T
    self_ = float(np.diag(BC).mean())
    # weights from A, table of i from B
    rho = float((T * BC).sum() / K)
    perm = rng.permutation(K)
    rho_null = float((T * (RA @ RB[perm].T)).sum() / K)
    d["self"], d["rho"], d["rho_null"] = self_, rho, rho_null
    d["eps_fam"] = 1 - rho / self_
    return d, T, Tb, BC


def spectral(Qs):
    """Qs: list of 4 K x K tables. eps_close from quarters."""
    def tilde(W):
        W = W / W.sum()
        a, b = W.sum(1), W.sum(0)
        return W / np.sqrt(np.outer(a, b) + 1e-300)
    t = [tilde(Offw(q)) for q in Qs]
    A = t[0] @ t[1].T
    Bm = t[2] @ t[3].T
    return float(1 - np.trace(A @ Bm) / np.trace(A)), float(np.trace(A)), float(np.trace(A @ Bm))


def Offw(q):
    w = q.copy()
    np.fill_diagonal(w, 0)
    return w


def eta_bounds(A, tries=40):
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)
    d = A.sum(1)
    ok = d > 0
    A, d = A[ok][:, ok], d[ok]
    N = A / np.sqrt(np.outer(d, d))
    lam = np.linalg.eigvalsh(N)
    best = 1.0
    tot = A.sum()
    for t in range(tries):
        x = rng.choice((-1., 1.), len(d))
        for _ in range(60):
            f = A @ x
            flip = np.where(x * f > 0)[0]
            if len(flip) == 0:
                break
            i = flip[np.argmax(np.abs(f[flip]))]
            x[i] = -x[i]
        best = min(best, float((A * (np.outer(x, x) > 0)).sum() / tot))
    return float((1 + lam[0]) / 2), best, lam[:3].tolist(), lam[-4:].tolist()


def capture(Y, lab, Wc, rs, held=True, iters=15):
    """Rows: photos. Y: N x C logits. lab: row -> class index. Wc: K x C metric."""
    K, C = Wc.shape
    Wc = Wc / Wc.sum(1, keepdims=True)
    idx = [np.where(lab == c)[0] for c in range(K)]
    D = np.zeros_like(Y)
    for c in range(K):
        D[idx[c]] = Y[idx[c]] - Y[idx[c]].mean(0)
    def energy(rows):
        tot = 0.
        for c in range(K):
            r = rows[c]
            d = D[r]
            dbar = d @ Wc[c]
            tot += (((d - dbar[:, None]) ** 2) * Wc[c][None, :]).sum()
        return tot
    fit_rows = [ix[::2] for ix in idx] if held else idx
    test_rows = [ix[1::2] for ix in idx] if held else idx
    Efit, Etest = energy(fit_rows), energy(test_rows)
    out = {}
    def codes(V, rows):
        Z = np.c_[V, np.ones(C)]
        cs = {}
        loss = 0.
        for c in range(K):
            d = D[rows[c]]
            w = Wc[c]
            G = (Z * w[:, None]).T @ Z + 1e-9 * np.eye(Z.shape[1])
            sol = np.linalg.solve(G, (Z * w[:, None]).T @ d.T).T
            res = d - sol @ Z.T
            loss += ((res ** 2) * w[None, :]).sum()
            cs[c] = sol
        return cs, loss
    for r in rs:
        # init: top directions of the metric-weighted movement
        samp = np.concatenate([D[fit_rows[c]] * np.sqrt(Wc[c])[None, :] for c in range(K)])
        U, s, Vt = np.linalg.svd(samp[rng.choice(len(samp), min(4000, len(samp)), replace=False)], full_matrices=False)
        V = Vt[:r].T * 1.0
        V = V + 1e-3 * rng.standard_normal(V.shape)
        for it in range(iters):
            cs, loss = codes(V, fit_rows)
            CC = np.zeros((K, r, r))
            CD = np.zeros((K, r, C))
            for c in range(K):
                cc = cs[c][:, :r]
                t = cs[c][:, r:]
                CC[c] = cc.T @ cc
                CD[c] = cc.T @ (D[fit_rows[c]] - t)
            A = np.einsum("kc,krs->crs", Wc, CC) + 1e-9 * np.eye(r)[None]
            b = np.einsum("kc,krc->cr", Wc, CD)
            V = np.linalg.solve(A, b[..., None])[..., 0]
        cs, loss = codes(V, fit_rows)
        cs2, loss2 = codes(V, test_rows)
        out[r] = (1 - loss / Efit, 1 - loss2 / Etest)
    return out


def movement_stats(Y, lab, Wc, fam_mask, side_mask, K):
    """fam_mask, side_mask: K x C bool, links of each row. Direction: first PC of sqrt(w) d."""
    C = Wc.shape[1]
    Wn = Wc / Wc.sum(1, keepdims=True)
    fr, sr, share = [], [], []
    for c in range(K):
        ix = np.where(lab == c)[0]
        d = Y[ix] - Y[ix].mean(0)
        w = Wn[c]
        d = d - (d @ w)[:, None]
        sq = np.sqrt(w)
        U, s, Vt = np.linalg.svd(d * sq[None, :], full_matrices=False)
        m = Vt[0] / np.where(sq > 1e-6, sq, np.inf)
        if abs(m[c]) < 1e-9:
            continue
        m = m / m[c]
        share.append(s[0] ** 2 / (s ** 2).sum())
        # relative to own: m[i] - 1 is M' of the own gauge
        if fam_mask[c].any():
            fr.append(np.average(m[fam_mask[c]], weights=w[fam_mask[c]]))
        if side_mask[c].any():
            sr.append(np.average(m[side_mask[c]], weights=w[side_mask[c]]))
    return float(np.median(fr)), float(np.median(sr)), float(np.mean(share))


res = {}
# ---- hard tables, in square form: own class = block column
cases = {"sq C5 t=1": (cycle(5), 1), "sq C5 t=3": (cycle(5), 3), "sq sub(K5) t=1": (subdivide(*complete(5)), 1), "sq sub(K5) t=2": (subdivide(*complete(5)), 2), "sq K5 t=2": (complete(5), 2)}
R = 6
cube = list(itertools.product((0, 1), repeat=R))
ci = {v: i for i, v in enumerate(cube)}
ed = set()
for v in cube:
    anti = tuple(1 - z for z in v)
    for j in range(R):
        u = list(anti)
        u[j] = 1 - u[j]
        a_, b_ = ci[v], ci[tuple(u)]
        if a_ != b_:
            ed.add((min(a_, b_), max(a_, b_)))
    a_, b_ = ci[v], ci[anti]
    ed.add((min(a_, b_), max(a_, b_)))
cases["sq noise cube R=6 t=1"] = ((len(cube), sorted(ed)), 1)
a_ = 1.0
for name, ((q, e), t) in cases.items():
    n = q * t
    D = float(np.exp(np.ceil(np.log(n ** 2 * 50))))
    M, W, fam, adj, Df = sqfam(q, e, t, a_, .3, D)
    y0 = np.log(W)
    Y = np.zeros((2 * n, n))
    Y[0::2] = y0 - a_ / 2 * M
    Y[1::2] = y0 + a_ / 2 * M
    lab = np.repeat(np.arange(n), 2)
    S = (softmax(Y[0::2]) + softmax(Y[1::2])) / 2
    d, T, Tb, BC = basic(S)
    fm = (fam[:, None] == fam[None, :]) & ~np.eye(n, dtype=bool)
    sm = adj[fam][:, fam]
    d["rho_fam"] = float((T * BC * fm).sum() / max((T * fm).sum(), 1e-300))
    d["rho_side"] = float((T * BC * sm).sum() / max((T * sm).sum(), 1e-300))
    d["eps_close"] = spectral([S, S, S, S])[0]
    Sb = S.copy()
    # link graph on families
    Af = np.zeros((q, q))
    Af[adj] = 1
    d["eta_lo"], d["eta_hi"], _, _ = eta_bounds(Af)
    d["capture"] = capture(Y, lab, S, [1, 2, 4], held=False)
    d["wins"] = float(len(set(softmax(Y).argmax(1).tolist())) / n)
    d["move_fam"], d["move_side"], d["move_share"] = movement_stats(Y, lab, S, fm, sm, n)
    res[name] = d
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}, flush=True)

# ---- original TQ31 table: map block column j to row j, drop g and b, keep filler in the mass
for name, (n, e) in {"TQ31 C5": cycle(5), "TQ31 sub(K5)": subdivide(*complete(5))}.items():
    D = float(np.exp(np.ceil(np.log(n ** 2 * 50))))
    y0, mv = tq31(n, e, 1.0, .31, D)
    sp = softmax(y0 + mv)
    block = sp[:, 1 + n:1 + 2 * n].copy()
    fill = sp[:, 1 + 2 * n:]
    # the map to a square form is only for the closure number: own = diag of the block
    d, T, _, BC = basic(block / block.sum(1, keepdims=True))
    res[name] = {k: d[k] for k in ("eps_fam", "rho", "self", "top1", "eff", "recip", "prod_rms", "level_rms")}
    res[name]["g_mass"] = float(sp[:, 0].min())
    res[name]["univ"] = float(sp[:, 0].min())
    print(name, {k: round(v, 3) for k, v in res[name].items()}, flush=True)

# ---- ResNet-18
Yall = np.load(f"{HERE}/cache/map/resnet18_y.npy").astype(np.float64)
meta = np.load(f"{HERE}/cache/map/resnet18_meta.npz")
lab = meta["labels"].astype(int)
Wout = meta["W"].astype(np.float64)
bout = meta["b"].astype(np.float64)
Hall = np.load(f"{HERE}/cache/map/resnet18_h.npy").astype(np.float64)
K = 1000
Sx = softmax(Yall)
idx = [np.where(lab == c)[0] for c in range(K)]
gen = np.random.default_rng(7)
parts = [np.zeros((K, K)) for _ in range(4)]
halfA, halfB = np.zeros((K, K)), np.zeros((K, K))
for c in range(K):
    ix = gen.permutation(idx[c])
    for q in range(4):
        parts[q][c] = Sx[ix[q * 5:(q + 1) * 5]].mean(0)
    halfA[c] = Sx[ix[:10]].mean(0)
    halfB[c] = Sx[ix[10:]].mean(0)
Sall = (halfA + halfB) / 2
d, T, Tb, BC = basic(Sall)
dh, Th, Thb, BCh = basic(halfA, halfB)
d["half_self"], d["half_rho"], d["half_rho_null"], d["half_eps_fam"] = dh["self"], dh["rho"], dh["rho_null"], dh["eps_fam"]
un = Wout / np.linalg.norm(Wout, axis=1, keepdims=True)
cosW = un @ un.T
head = np.argsort(-Th, 1)[:, :8]
hm = np.zeros((K, K), bool)
hm[np.arange(K)[:, None], head] = True
fm = hm & (cosW >= .3)
sm = hm & (cosW < .1)
d["head_family_share"] = float((Th * fm).sum() / K)
d["head_side_share"] = float((Th * sm).sum() / K)
d["rho_fam"] = float((Th * BCh * fm).sum() / (Th * fm).sum())
d["rho_side"] = float((Th * BCh * sm).sum() / (Th * sm).sum())
d["rho_fam_rel"], d["rho_side_rel"] = d["rho_fam"] / dh["self"], d["rho_side"] / dh["self"]
# noise of the log table between halves, on A's weights
w = Th / K
d["half_rms"] = float(np.sqrt((w * (np.log(Th + 1e-12) - np.log(Thb + 1e-12)) ** 2).sum()))
d["eps_close"], d["tr2"], d["tr4"] = spectral(parts)
d["wins"] = float(len(set(Yall.argmax(1).tolist())) / K)
d["wins_own"] = float((Yall.argmax(1) == lab).mean())
# eta on all head links and on families contracted
At = np.zeros((K, K))
top = np.argsort(-T, 1)[:, :8]
At[np.arange(K)[:, None], top] = T[np.arange(K)[:, None], top]
d["eta_lo"], d["eta_hi"], lo3, hi4 = eta_bounds(At.copy())
print("resnet link graph lambda low", lo3, "high", hi4, flush=True)
from scipy.cluster.hierarchy import fcluster, linkage
Zl = linkage(un, "ward")
fams = fcluster(Zl, 120, "maxclust") - 1
qq = fams.max() + 1
Af = np.zeros((qq, qq))
for c in range(K):
    for i in top[c]:
        if fams[c] != fams[i]:
            Af[fams[c], fams[i]] += T[c, i]
d["eta_lo_fam"], d["eta_hi_fam"], lo3, hi4 = eta_bounds(Af)
print("resnet family graph lambda low", lo3, "high", hi4, flush=True)
d["in_family_mass"] = float(sum(T[c, top[c]][fams[top[c]] == fams[c]].sum() for c in range(K)) / K)
d["capture"] = capture(Yall, lab, Sall, [1, 2, 4, 8, 14])
d["move_fam"], d["move_side"], d["move_share"] = movement_stats(Yall, lab, Sall, fm, sm, K)
res["ResNet-18"] = d
print("ResNet-18", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}, flush=True)

# ---- D42 and D45 at the last hidden layer on photo rows, by sampling
Wt = Wout - Wout.mean(0, keepdims=True)
N, U_ = Hall.shape
mass = Hall.mean(0)
live = np.where(mass > 1e-9)[0]
om = mass[live] * np.linalg.norm(Wt[:, live], axis=0)
units = gen.choice(live, 120, replace=False, p=om / om.sum())
Vs, Vr, lam, xi = [], [], [], []
for u in units:
    mu = Hall[:, u] / Hall[:, u].sum()
    xs = gen.choice(N, 40, p=mu)
    # readers: pairs with probability proportional to |W_cu - W_c'u|, 40 000 candidates
    c1, c2 = gen.integers(0, K, 40000), gen.integers(0, K, 40000)
    keep = c1 != c2
    c1, c2 = c1[keep], c2[keep]
    wr = np.abs(Wout[c1, u] - Wout[c2, u])
    pk = gen.choice(len(c1), 300, p=wr / wr.sum())
    vot = (Wout[c1[pk]] - Wout[c2[pk]])[None, :, :] * Hall[xs][:, None, :]
    s = np.abs(vot).sum(-1)
    t = np.abs(vot.sum(-1))
    Vs.append(float(((s - t) / s).mean()))
    # relevant pairs: at x, sample pairs by q_c q_c' |W_cu - W_c'u|
    vv = []
    for x in xs[:15]:
        q = Sx[x]
        pp = gen.choice(K, 400, p=q)
        pp2 = gen.choice(K, 400, p=q)
        ok = pp != pp2
        wg = np.abs(Wout[pp[ok], u] - Wout[pp2[ok], u])
        if wg.sum() == 0:
            continue
        v = (Wout[pp[ok]] - Wout[pp2[ok]]) * Hall[x][None, :]
        k = (np.abs(v).sum(1) - np.abs(v.sum(1))) / np.abs(v).sum(1)
        vv.append(float((wg * k).sum() / wg.sum()))
    Vr.append(np.mean(vv))
    # lambda and xi
    S_all = Sx @ Wt[:, u]
    l = float(mu @ S_all - S_all.mean()) / np.abs(Wt[:, u]).sum()
    lam.append(l)
    xt = gen.choice(N, 300, p=mu)
    xn = gen.integers(0, N, 300)
    xr = gen.integers(0, N, 300)
    def Sind(xa):
        Rr = Yall[xr] - Wout[:, u][None, :] * Hall[xr, u][:, None]
        return softmax(Wout[:, u][None, :] * Hall[xa, u][:, None] + Rr) @ Wt[:, u]
    li = float(Sind(xt).mean() - Sind(xn).mean()) / np.abs(Wt[:, u]).sum()
    xi.append(l - li)
lamv, xiv = np.array(lam), np.array(xi)
d2 = dict(eps_V=float(np.mean(Vs)), V_rel=float(np.mean(Vr)), lam_pos=float((lamv > 0).mean()), xi_pos=float((xiv > 0).mean()), xi_median=float(np.median(xiv)), active=float((Hall > 1e-6).mean()))
res["ResNet-18 D42 D45"] = d2
print("ResNet-18 last layer (omega-weighted sample of units):", d2, flush=True)
json.dump(res, open(f"{HERE}/rounds/round4/i5/d_compare.json", "w"), indent=1, default=lambda z: {str(k): v for k, v in z.items()} if isinstance(z, dict) else float(z))
