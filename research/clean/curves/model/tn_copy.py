# serves: TN1
"""Exact check of Propositions TN1, TN2 and Theorem TN3 (COARSE-THEORY.md B.N) on random instances.

Instance: one hidden layer, three logits, 8 affinely independent points in R^8 (so every table of values is relu of an
affine function and the units are real network units), random point weights p.
  R1 = {0, 1} meets the group S = {g1, g2} (box of size 1, relu(m . z_S + beta), fixed); exact district and shelter.
  k1, k2: kept, district with region R1, judgment claim (R1, {k1, k2}, a = (1, 1/2)) with error; k1 leaks at point 5.
  R2 = {2, 3, 4} meets no group: district {f1, f2}; f1 leaks at point 6.  Points 5, 6, 7 lie in no region.
  q: kept, flat claim (constant on R1, noisy on R2).
Teacher: N's own logits with random weights. Builds: the fixed box (lifted to copies) and a free head on every kept
unit, the box unit and a bias. All values below are per fixed box: the lift and merge maps of TN1 send fixed boxes to
fixed boxes, so the equalities of TN1 part 3 hold box by box.
V_rob: the loss is a sum over points and the entries of different points move independently, so the worst case is
the sum over x of p_x times the largest KL over the vertices of x's own entries (TE3 part 3); minimised in epigraph
form with SLSQP.
Checks (C1-C8), for copies (n = 2, 3, random shares) of k1, f1, q, g1, and for splits (TN2) of k1 and f1:
  C1 copy: y_N' = y_N on 200 random inputs off P.
  C2 copy: the clean table of the copy is lam_i times u's (DE2 v3 shares); the equal-split rule of DE2 v2 is not.
  C3 copy: V_b, V_rob and the clean-table minimum equal N's (to the solver's accuracy).
  C4 copy: E of the lifted build equals E of the build; E of the merge of a random build of N' is at most its E.
  C5 copy: every error of the map is unchanged in the 2026-10-28 forms (largest member's term for district, shelter
     and flat claims), and the class count c(S) of DM6 item 5 is unchanged; the first, summed forms of eps_dist and
     eps_shel grow by (n - 1) eps_u and (n - 1) e_u (shelter on a variant with a deviation outside R1).
  C6 copy and split: M, O, phi, D_M, Lambda_M do not change.
  C7 split: V_b and V_rob do not rise; the district error (largest term) rises or stays, and is at most 1.
  C8 TN3: terms (a) + (b) of E equal M_K phi_K + M_S sigma, term (c) equals D°_M; Lambda_M <= rho phi; rho >= 1.
"""
import json
import os

import numpy as np
from scipy.optimize import minimize

NY, NP, DIM = 3, 8, 8
R1, R2 = [0, 1], [2, 3, 4]
NOREG = [5, 6, 7]
TOL = 1e-6


def lse(Y):
    m = Y.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(Y - m).sum(axis=-1, keepdims=True)))[..., 0]


def softmax(Y):
    return np.exp(Y - lse(Y)[..., None])


def osc(v):
    return float(np.max(v) - np.min(v))


def make_instance(rng):
    X = rng.normal(size=(NP, DIM))
    p = rng.uniform(0.5, 1.5, NP); p /= p.sum()
    Z = {}
    out = [x for x in range(NP) if x not in R1]
    for g in ["g1", "g2"]:
        z = np.full(NP, -rng.uniform(0.5, 1.5)); z[R1] = rng.uniform(0.3, 1.5, len(R1)); Z[g] = z
    for k in ["k1", "k2"]:
        z = np.full(NP, -1.0); z[R1] = rng.uniform(0.4, 1.4, len(R1)); Z[k] = z
    Z["k1"][5] = rng.uniform(0.05, 0.2)                      # k1 leaks at point 5
    for f in ["f1", "f2"]:
        z = np.full(NP, -1.0); z[R2] = rng.uniform(0.3, 1.5, len(R2)); Z[f] = z
    Z["f1"][6] = rng.uniform(0.05, 0.2)                      # f1 leaks at point 6
    z = np.full(NP, 1.0); z[R2] = 1.0 + rng.uniform(-0.2, 0.2, len(R2)); z[NOREG] = rng.uniform(0.5, 1.5, 3); Z["q"] = z
    units = ["g1", "g2", "k1", "k2", "f1", "f2", "q"]
    net = dict(X=X, p=p, units=units, Z=Z, W={u: rng.normal(size=NY) for u in units}, c=rng.normal(size=NY))
    net["aff"] = {u: affine_for(X, Z[u]) for u in units}
    mp = dict(dist=[(["g1", "g2"], R1), (["k1", "k2"], R1), (["f1", "f2"], R2)],
              shel=[0], judg=(R1, ["k1", "k2"], {"k1": 1.0, "k2": 0.5}), flat=["q"], group=["g1", "g2"])
    box = dict(m={"g1": rng.normal(), "g2": rng.normal()}, beta=rng.normal())
    return net, mp, box


def affine_for(X, z):
    A = np.hstack([X, np.ones((len(X), 1))])
    sol = np.linalg.lstsq(A, z, rcond=None)[0]
    assert np.allclose(A @ sol, z, atol=1e-10)
    return sol


def H(net):
    return {u: np.maximum(net["Z"][u], 0.0) for u in net["units"]}


def logits_N(net, h=None):
    h = H(net) if h is None else h
    return sum(np.outer(h[u], net["W"][u]) for u in net["units"]) + net["c"]


def logits_off_P(net, Xr):
    A = np.hstack([Xr, np.ones((len(Xr), 1))])
    return sum(np.outer(np.maximum(A @ net["aff"][u], 0), net["W"][u]) for u in net["units"]) + net["c"]


def kept(net, mp):
    return [u for u in net["units"] if u not in mp["group"]]


def clean_table(net, mp, rule="v3"):
    p, h = net["p"], H(net)
    h0 = {u: h[u].copy() for u in net["units"]}
    z0 = {u: net["Z"][u].copy() for u in mp["group"]}
    for D, Om in mp["dist"]:
        out = [x for x in range(NP) if x not in Om]
        for u in D:
            h0[u][out] = 0.0
    for i in mp["shel"]:
        D, Om = mp["dist"][i]
        out = [x for x in range(NP) if x not in Om]
        for u in D:
            if u in z0:
                z0[u][out] = (p[out] * net["Z"][u][out]).sum() / p[out].sum()
    for u in mp["flat"]:
        for R in (R1, R2):
            h0[u][R] = (p[R] * h[u][R]).sum() / p[R].sum()
    R, K, a = mp["judg"]
    Kp = [k for k in K if a[k] != 0]
    s = sum(a[k] * h[k][R] for k in K)
    tot = sum(abs(a[k]) * (p[R] * h[k][R]).sum() for k in Kp)
    for k in Kp:
        om = (abs(a[k]) * (p[R] * h[k][R]).sum() / tot) if (rule == "v3" and tot > 0) else 1.0 / len(Kp)
        h0[k][R] = h[k][R] + om * (1 - s) / a[k]
    return h0, z0


def box_values(net, box, z):
    return np.maximum(sum(box["m"][u] * z[u] for u in box["m"]) + box["beta"], 0.0)


def features(net, mp, box, h, z):
    ks = kept(net, mp)
    return np.column_stack([h[u] for u in ks] + [box_values(net, box, z), np.ones(NP)])


def kl_pts(Q, Y):
    return (Q * np.log(Q)).sum(-1) - (Q * Y).sum(-1) + lse(Y)


def V_b(net, mp, box, h, z, Q):
    F = features(net, mp, box, h, z); p = net["p"]; nf = F.shape[1]

    def f(t):
        T = t.reshape(nf, NY); Y = F @ T; S = softmax(Y)
        val = (p * kl_pts(Q, Y)).sum()
        return val, (F.T @ (p[:, None] * (S - Q))).ravel()
    best = min((minimize(f, np.random.default_rng(s).normal(0, .1, nf * NY), jac=True, method="L-BFGS-B",
                         options=dict(maxiter=20000, ftol=1e-15, gtol=1e-12)) for s in range(3)), key=lambda r: r.fun)
    return best.fun, best.x.reshape(nf, NY)


def vertex_tables(net, mp, box):
    """Per point x: the list of feature rows at the vertices of x's own moved entries (no shelter entry moves here)."""
    h, (h0, z0) = H(net), clean_table(net, mp)
    ks = kept(net, mp)
    C = box_values(net, box, net["Z"])
    rows = []
    for x in range(NP):
        mv = [j for j, u in enumerate(ks) if abs(h[u][x] - h0[u][x]) > 1e-14]
        base = np.array([h[u][x] for u in ks] + [C[x], 1.0])
        V = []
        for bits in range(2 ** len(mv)):
            r = base.copy()
            for b, j in enumerate(mv):
                if bits >> b & 1:
                    r[j] = h0[ks[j]][x]
            V.append(r)
        rows.append(np.array(V))
    return rows


def V_rob(net, mp, box, Q, T0):
    rows = vertex_tables(net, mp, box); p = net["p"]; nf = rows[0].shape[1]
    nT = nf * NY

    def worst(T):
        return sum(p[x] * kl_pts(np.repeat(Q[x][None], len(rows[x]), 0), rows[x] @ T).max() for x in range(NP))

    def obj(v):
        return (p * v[nT:]).sum()

    def objg(v):
        g = np.zeros_like(v); g[nT:] = p; return g

    def cons(v):
        T = v[:nT].reshape(nf, NY)
        return np.concatenate([v[nT + x] - kl_pts(np.repeat(Q[x][None], len(rows[x]), 0), rows[x] @ T)
                               for x in range(NP)])

    def consj(v):
        T = v[:nT].reshape(nf, NY); J = []
        for x in range(NP):
            Y = rows[x] @ T; S = softmax(Y)
            for r, srow in zip(rows[x], S):
                g = np.zeros_like(v)
                g[:nT] = -np.outer(r, srow - Q[x]).ravel(); g[nT + x] = 1.0
                J.append(g)
        return np.array(J)
    best = None
    for T in [T0, T0 + np.random.default_rng(7).normal(0, .05, T0.shape)]:
        tx = np.array([kl_pts(np.repeat(Q[x][None], len(rows[x]), 0), rows[x] @ T).max() for x in range(NP)])
        v0 = np.concatenate([T.ravel(), tx + 1e-6])
        r = minimize(obj, v0, jac=objg, constraints=[dict(type="ineq", fun=cons, jac=consj)], method="SLSQP",
                     options=dict(maxiter=2000, ftol=1e-14))
        val = worst(r.x[:nT].reshape(nf, NY))
        if best is None or val < best[0]:
            best = (val, r.x[:nT].reshape(nf, NY))
    return best


def E_terms(net, mp, box, T, group_regions=(R1,)):
    """TE4's three terms for the build with head T (swings from T), N's swings in term (c)."""
    h, (h0, z0), p = H(net), clean_table(net, mp), net["p"]
    ks = kept(net, mp)
    sw = {u: osc(T[j]) for j, u in enumerate(ks)}
    swbox = osc(T[len(ks)])
    ta = sum(sw[u] * (p * np.abs(h[u] - h0[u])).sum() for u in ks)
    tb = sum(swbox * abs(box["m"][u]) * (p * np.abs(net["Z"][u] - z0[u])).sum() for u in mp["group"])
    nog = [x for x in range(NP) if not any(x in R for R in group_regions)]
    tc = sum(p[x] * osc(sum(net["W"][u] * (h[u][x] - h0[u][x]) for u in net["units"])) for x in nog)
    return ta, tb, tc


def errors(net, mp, zshift=None):
    """eps_dist, eps_shel (with an optional deviation added to the members' z outside R1), eps_judg, eps_quiet, eps_flat."""
    h, p = H(net), net["p"]
    Z = {u: net["Z"][u] + (zshift[u] if zshift and u in zshift else 0) for u in net["units"]}
    out = {}
    for i, (D, Om) in enumerate(mp["dist"]):
        o = [x for x in range(NP) if x not in Om]
        terms = [(p[o] * h[u][o]).sum() / (p * h[u]).sum() for u in D if (p * h[u]).sum() > 0]
        out[f"dist{i}"] = sum(terms)                       # first form (sum over members)
        out[f"distmax{i}"] = max(terms) if terms else 0.0   # 2026-10-28 form (largest member's term)
        out[f"quiet{i}"] = sum(p[x] * osc(sum(net["W"][u] * h[u][x] for u in D)) for x in o)
    D, Om = mp["dist"][0]; o = [x for x in range(NP) if x not in Om]
    es = []
    for u in D:
        zb = (p * Z[u]).sum(); Bu = (p * (Z[u] - zb) ** 2).sum()
        if Bu > 0:
            es.append(sum(p[x] * p[y] * (Z[u][x] - Z[u][y]) ** 2 for x in o for y in o) / Bu)
    out["shel"] = sum(es); out["shelmax"] = max(es) if es else 0.0
    zs = [Z[u] for u in mp["group"]]
    cls = []
    for z in zs:
        if not any(np.allclose(z * (np.abs(c).max() / np.abs(z).max()), c) for c in cls):
            cls.append(z)
    out["classes"] = len(cls)
    R, K, a = mp["judg"]
    out["judg"] = float((p[R] * (1 - sum(a[k] * h[k][R] for k in K)) ** 2).sum())
    for u in mp["flat"]:
        hb = (p * h[u]).sum(); Vu = (p * (h[u] - hb) ** 2).sum()
        out[f"flat_{u}"] = sum(p[x] * p[y] * (h[u][x] - h[u][y]) ** 2 for R in (R1, R2) for x in R for y in R) / Vu
    out["flatmax"] = max(out[f"flat_{u}"] for u in mp["flat"])
    return out


def dn2(net, mp, group_regions=(R1,)):
    h, (h0, _), p = H(net), clean_table(net, mp), net["p"]
    mu = {u: osc(net["W"][u]) * (p * h[u]).sum() for u in net["units"]}
    d = {u: (p * np.abs(h[u] - h0[u])).sum() for u in net["units"]}
    M = sum(mu.values())
    phi = sum(osc(net["W"][u]) * d[u] for u in net["units"]) / M
    O = sum(p[x] * osc(sum(net["W"][u] * h[u][x] for u in net["units"])) for x in range(NP))
    DM = sum(p[x] * osc(sum(net["W"][u] * (h[u][x] - h0[u][x]) for u in net["units"])) for x in range(NP))
    nog = [x for x in range(NP) if not any(x in R for R in group_regions)]
    Dd = sum(p[x] * osc(sum(net["W"][u] * (h[u][x] - h0[u][x]) for u in net["units"])) for x in nog)
    return dict(M=M, O=O, phi=phi, D_M=DM, Lambda=DM / O, rho=M / O, D_deg=Dd)


def masses(net, mp, box, T):
    h, (h0, z0), p = H(net), clean_table(net, mp), net["p"]
    ks = kept(net, mp)
    sw = {u: osc(T[j]) for j, u in enumerate(ks)}; swbox = osc(T[len(ks)])
    A = {u: (p * h[u]).sum() for u in ks}
    MK = sum(sw[u] * A[u] for u in ks if A[u] > 0)
    phiK = sum(sw[u] * (p * np.abs(h[u] - h0[u])).sum() for u in ks if A[u] > 0) / MK
    nu, sig = {}, {}
    for u in mp["group"]:
        zb = (p * net["Z"][u]).sum(); Bu = (p * (net["Z"][u] - zb) ** 2).sum()
        nu[u] = swbox * abs(box["m"][u]) * np.sqrt(Bu)
        sig[u] = (p * np.abs(net["Z"][u] - z0[u])).sum() / np.sqrt(Bu)
    MS = sum(nu.values())
    sigma = sum(nu[u] * sig[u] for u in nu) / MS if MS > 0 else 0.0
    return MK, phiK, MS, sigma


def copy_of(net, mp, box, u, lams):
    new = [f"{u}#{i}" for i in range(len(lams))]
    N2 = dict(X=net["X"], p=net["p"], c=net["c"], units=[], Z={}, W={}, aff={})
    for v in net["units"]:
        if v == u:
            for nm, l in zip(new, lams):
                N2["units"].append(nm); N2["Z"][nm] = l * net["Z"][u]; N2["W"][nm] = net["W"][u].copy()
                N2["aff"][nm] = l * net["aff"][u]
        else:
            N2["units"].append(v); N2["Z"][v] = net["Z"][v]; N2["W"][v] = net["W"][v]; N2["aff"][v] = net["aff"][v]
    return N2, rename_map(mp, u, new), lift_box(box, u, new)


def split_of(net, mp, box, u, parts):
    """TN2: narrow units with values parts[i] (>= 0, adding to h_u on P), each relu of an affine map."""
    new = [f"{u}~{i}" for i in range(len(parts))]
    N2 = dict(X=net["X"], p=net["p"], c=net["c"], units=[], Z={}, W={}, aff={})
    for v in net["units"]:
        if v == u:
            for nm, hv in zip(new, parts):
                z = np.where(hv > 0, hv, -1.0)
                N2["units"].append(nm); N2["Z"][nm] = z; N2["W"][nm] = net["W"][u].copy()
                N2["aff"][nm] = affine_for(net["X"], z)
        else:
            N2["units"].append(v); N2["Z"][v] = net["Z"][v]; N2["W"][v] = net["W"][v]; N2["aff"][v] = net["aff"][v]
    return N2, rename_map(mp, u, new), lift_box(box, u, new)


def rename_map(mp, u, new):
    rep = lambda L: [w for v in L for w in (new if v == u else [v])]
    R, K, a = mp["judg"]
    a2 = {(w): a[v] for v in K for w in (new if v == u else [v])}
    return dict(dist=[(rep(D), Om) for D, Om in mp["dist"]], shel=mp["shel"], judg=(R, rep(K), a2),
                flat=rep(mp["flat"]), group=rep(mp["group"]))


def lift_box(box, u, new):
    m = {}
    for v, w in box["m"].items():
        for nm in (new if v == u else [v]):
            m[nm] = w
    return dict(m=m, beta=box["beta"])


def lift_head(net, mp, T, net2, mp2, u):
    ks, ks2 = kept(net, mp), kept(net2, mp2)
    rows = []
    for v in ks2:
        base = v.split("#")[0].split("~")[0]
        rows.append(T[ks.index(base)])
    return np.vstack(rows + [T[len(ks)], T[len(ks) + 1]])


def merge_head(net, mp, T2, net2, mp2, u, lams):
    ks, ks2 = kept(net, mp), kept(net2, mp2)
    rows = []
    for v in ks:
        if v == u:
            rows.append(sum(l * T2[ks2.index(f"{u}#{i}")] for i, l in enumerate(lams)))
        else:
            rows.append(T2[ks2.index(v)])
    return np.vstack(rows + [T2[len(ks2)], T2[len(ks2) + 1]])


def run_instance(seed, log):
    rng = np.random.default_rng(seed)
    net, mp, box = make_instance(rng)
    Q = softmax(logits_N(net))
    h0, z0 = clean_table(net, mp)
    vb, Tb = V_b(net, mp, box, H(net), net["Z"], Q)
    vc, Tc = V_b(net, mp, box, h0, z0, Q)
    vr, Tr = V_rob(net, mp, box, Q, Tb)
    base = dict(V_b=vb, V_clean=vc, V_rob=vr, **dn2(net, mp))
    err = errors(net, mp)
    E0 = E_terms(net, mp, box, Tc)
    ok = vc <= vr + TOL and vb <= vr + TOL
    res = dict(seed=seed, base=base, cases=[])
    # C8 on N
    MK, phiK, MS, sig = masses(net, mp, box, Tc)
    c8 = (abs(E0[0] + E0[1] - (MK * phiK + MS * sig)) < 1e-10 and abs(E0[2] - base["D_deg"]) < 1e-10
          and base["Lambda"] <= base["rho"] * base["phi"] + 1e-12 and base["rho"] >= 1 - 1e-12)
    res["C8"] = c8; ok &= c8
    for u in ["k1", "f1", "q", "g1"]:
        for n in (2, 3):
            lams = rng.dirichlet(np.ones(n))
            N2, M2, B2 = copy_of(net, mp, box, u, lams)
            Xr = rng.normal(size=(200, DIM))
            c1 = np.abs(logits_off_P(N2, Xr) - logits_off_P(net, Xr)).max() < 1e-10
            h02, z02 = clean_table(N2, M2)
            c2 = all(np.allclose(h02[f"{u}#{i}"], l * h0[u], atol=1e-12) for i, l in enumerate(lams)) and \
                all(np.allclose(h02[v], h0[v], atol=1e-12) for v in net["units"] if v != u)
            old_gap = None
            if u == "k1":
                h02o, _ = clean_table(N2, M2, rule="v2")
                old_gap = float(max(np.abs(h02o[f"{u}#{i}"] - l * h0[u]).max() for i, l in enumerate(lams)))
            Q2 = softmax(logits_N(N2))
            vb2, Tb2 = V_b(N2, M2, B2, H(N2), N2["Z"], Q2)
            vc2, _ = V_b(N2, M2, B2, h02, z02, Q2)
            vr2, Tr2 = V_rob(N2, M2, B2, Q2, lift_head(net, mp, Tb, N2, M2, u))
            c3 = abs(vb2 - vb) < TOL and abs(vc2 - vc) < TOL and abs(vr2 - vr) < TOL
            Tl = lift_head(net, mp, Tc, N2, M2, u)
            El = E_terms(N2, M2, B2, Tl)
            Trand = rng.normal(size=Tl.shape)
            Er = E_terms(N2, M2, B2, Trand)
            Em = E_terms(net, mp, box, merge_head(net, mp, Trand, N2, M2, u, lams)) if u != "g1" else None
            c4 = np.allclose(El, E0, atol=1e-12) and (Em is None or sum(Em) <= sum(Er) + 1e-12)
            err2 = errors(N2, M2)
            i_d = [i for i, (D, _) in enumerate(mp["dist"]) if u in D]
            epsu = None
            c5 = abs(err2["judg"] - err["judg"]) < 1e-12 and all(abs(err2[f"quiet{i}"] - err[f"quiet{i}"]) < 1e-12
                                                               for i in range(3))
            if i_d:
                i = i_d[0]; D, Om = mp["dist"][i]; o = [x for x in range(NP) if x not in Om]
                hu = H(net)[u]; epsu = (net["p"][o] * hu[o]).sum() / (net["p"] * hu).sum()
                c5 &= abs(err2[f"dist{i}"] - err[f"dist{i}"] - (n - 1) * epsu) < 1e-12      # first form grows
                c5 &= abs(err2[f"distmax{i}"] - err[f"distmax{i}"]) < 1e-12                  # new form does not
            c5 &= err2["classes"] == err["classes"] and abs(err2["flatmax"] - err["flatmax"]) < 1e-12
            if u == "q":
                c5 &= all(abs(err2[f"flat_q#{i}"] - err["flat_q"]) < 1e-12 for i in range(n))
            if u == "g1":   # shelter count on a variant with a deviation of g1's pre-activation outside R1
                zs = {"g1": np.where(np.isin(np.arange(NP), R1), 0.0, rng.normal(0, .1, NP))}
                zs2 = {f"g1#{i}": l * zs["g1"] for i, l in enumerate(lams)}
                e1, e2 = errors(net, mp, zs)["shel"], errors(N2, M2, zs2)["shel"]
                Z = net["Z"]["g1"] + zs["g1"]; p = net["p"]; zb = (p * Z).sum(); Bu = (p * (Z - zb) ** 2).sum()
                o = [x for x in range(NP) if x not in R1]
                eu = sum(p[x] * p[y] * (Z[x] - Z[y]) ** 2 for x in o for y in o) / Bu
                c5 &= abs(e2 - e1 - (n - 1) * eu) < 1e-12
                c5 &= abs(errors(N2, M2, zs2)["shelmax"] - errors(net, mp, zs)["shelmax"]) < 1e-12
            d2 = dn2(N2, M2)
            c6 = all(abs(d2[k] - base[k]) < 1e-12 for k in ("M", "O", "phi", "D_M", "Lambda"))
            case = dict(kind="copy", u=u, n=n, lams=lams.tolist(), C1=bool(c1), C2=bool(c2), C3=bool(c3), C4=bool(c4),
                        C5=bool(c5), C6=bool(c6), dV_b=vb2 - vb, dV_clean=vc2 - vc, dV_rob=vr2 - vr,
                        E_lift=float(sum(El)), E=float(sum(E0)), old_rule_gap=old_gap)
            res["cases"].append(case); ok &= c1 and c2 and c3 and c4 and c5 and c6
    for u in ["k1", "f1"]:
        for n in (2, 3):
            hu = H(net)[u]
            Wt = rng.dirichlet(np.ones(n), size=NP).T
            parts = [w * hu for w in Wt]
            N2, M2, B2 = split_of(net, mp, box, u, parts)
            Q2 = softmax(logits_N(N2))
            c1 = np.abs(logits_N(N2) - logits_N(net)).max() < 1e-10
            vb2, Tb2 = V_b(N2, M2, B2, H(N2), N2["Z"], Q2)
            vr2, _ = V_rob(N2, M2, B2, Q2, lift_head(net, mp, Tb, N2, M2, u))
            c7 = vb2 <= vb + TOL and vr2 <= vr + TOL
            d2 = dn2(N2, M2)
            c6 = all(abs(d2[k] - base[k]) < 1e-12 for k in ("M", "O", "phi", "D_M", "Lambda"))
            err2 = errors(N2, M2)
            i = [i for i, (D, _) in enumerate(mp["dist"]) if u in D][0]
            grow = err2[f"distmax{i}"] - err[f"distmax{i}"]
            c7 &= -1e-12 <= grow and err2[f"distmax{i}"] <= 1 + 1e-12
            case = dict(kind="split", u=u, n=n, C1=bool(c1), C6=bool(c6), C7=bool(c7), dV_b=vb2 - vb, dV_rob=vr2 - vr,
                        dist_growth=grow)
            res["cases"].append(case); ok &= c1 and c6 and c7
    res["ok"] = bool(ok)
    print(f"seed {seed}: V_b {vb:.6f} V_clean {vc:.6f} V_rob {vr:.6f} | phi {base['phi']:.4f} Lambda "
          f"{base['Lambda']:.4f} rho {base['rho']:.3f} | C8 {c8} | {'ok' if ok else 'BROKEN'}", flush=True)
    for c in res["cases"]:
        if not all(v for k, v in c.items() if k.startswith("C")):
            print("   FAIL", c, flush=True)
    worst = max(max(abs(c["dV_b"]), abs(c.get("dV_rob", 0)), abs(c.get("dV_clean", 0))) for c in res["cases"]
                if c["kind"] == "copy")
    gaps = [c["old_rule_gap"] for c in res["cases"] if c.get("old_rule_gap") is not None]
    drops = [c["dV_rob"] for c in res["cases"] if c["kind"] == "split"]
    print(f"   copies: largest |change| of V_b, V_clean, V_rob {worst:.2e}; equal-split rule (DE2 v2) misses the copy "
          f"by up to {max(gaps):.3e}; splits: V_rob changes {min(drops):+.2e} .. {max(drops):+.2e}", flush=True)
    log.append(res)
    return ok


if __name__ == "__main__":
    log = []
    oks = [run_instance(s, log) for s in range(20261101, 20261109)]
    print(f"{sum(oks)} of {len(oks)} instances pass every check")
    with open(os.path.join(os.path.dirname(__file__), "tn_copy.json"), "w") as f:
        json.dump(log, f, indent=1, default=float)
