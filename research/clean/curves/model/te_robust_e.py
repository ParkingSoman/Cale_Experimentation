# serves: TE4
"""Exact check of Theorem TE4 (COARSE-THEORY.md B.E) on random instances with three logits.

Instance (12 points): R1 = 0..3 meets group S1 = {u1, u2} (size 0, dropped); R2 = 4..7 meets S2 = {u3, u4, u5} (a fixed
size-1 box relu(m.z + beta)); R3 = 8..9 meets no group; points 10, 11 are in no region. Kept units: k1, k2 (judgments
of R1, R2), e1, e2 (in R1's and R2's districts), f1, f2 (R3's district), g (flat: one value on each region).
N's teacher: random logit weights on every unit.
One fixed perturbation direction D per instance, scaled by eps, built so that the clean table of Definition DE2 is the
same for every eps: kept-unit leaks at points outside their regions (clean value 0), u1 at the threshold outside its
region (pre-activation 0) plus zero-mean noise (so it leaks with values ~eps), zero-mean shelter noise for the other
members, zero-mean flat noise for g, and judgment noise.
Checks, for each eps:
  (i)   eps = 0: B^0's loss equals the sum of L^0_S (the identity of TM1's construction),
  (Definition DE2 v2: judgment entries have a clean value at which a.h = 1 (h_k + (1 - s) / (|K'| a_k)); term (c) of the first
  form of TE4 is then part of term (a).)
  (DE2 v3, 2026-10-28, splits the move by shares omega_k in place of 1 / |K'|. Every judgment here has one unit, so
  omega_k = 1 and the results are unchanged.)
  (ii)  sum L^0 <= min found of the worst-case loss over U (fixed boxes, free head) <= sup_U D(B^0) <= sum L^0 + E,
        and D(B^0) on N's own table <= sup_U D(B^0),
  (iii) E / eps stays bounded as eps -> 0 (the terms of E are linear in the perturbation),
  (iv)  sum L^0 - V_b(N) (fixed boxes, N's own table) need not vanish as eps -> 0 (Proposition TE1).
The robust minimum is searched with a smoothed max (log-sum-exp at growing sharpness, L-BFGS), from B^0's head and
from the head fitted on N's table; the reported value is the true worst case at the best head found, so it is an
upper bound on the robust minimum with these boxes. Check (ii)'s left inequality then tests that no head found beats
the clean split, which is what TE4's lower bound says.
"""
import itertools
import json
import os

import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(20261029)
NY = 3
KEPT = ["k1", "e1", "k2", "e2", "f1", "f2", "g"]
MEMB = ["u1", "u2", "u3", "u4", "u5"]
HOME = {"u1": 1, "u2": 1, "u3": 2, "u4": 2, "u5": 2, "k1": 1, "e1": 1, "k2": 2, "e2": 2, "f1": 3, "f2": 3}


def lse(Y):
    m = Y.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(Y - m).sum(axis=-1, keepdims=True)))[..., 0]


def softmax(Y):
    return np.exp(Y - lse(Y)[..., None])


def kl_rows(Q, Y):
    return (Q * np.log(Q)).sum(-1) - (Q * Y).sum(-1) + lse(Y)


def osc(v):
    return float(np.max(v) - np.min(v))


def fit_head(Q, F, w, T0=None):
    nf = F.shape[1]

    def f(t):
        T = t.reshape(nf, NY); Y = F @ T
        return float(w @ kl_rows(Q, Y)), (F.T @ (w[:, None] * (softmax(Y) - Q))).ravel()
    r = minimize(f, np.zeros(nf * NY) if T0 is None else T0.ravel(), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=20000, ftol=1e-16, gtol=1e-11))
    return float(r.fun), r.x.reshape(nf, NY)


def worst_case(Q, rows, pt, w, T):
    """sum_x w_x max over the vertex rows of point x of KL(Q_x || softmax(row T))."""
    L = kl_rows(Q[pt], rows @ T)
    return float(sum(w[x] * L[pt == x].max() for x in np.unique(pt)))


def robust_min(Q, rows, pt, w, starts):
    nf = rows.shape[1]
    best = min(worst_case(Q, rows, pt, w, T) for T in starts)
    for T0 in starts:
        t = T0.ravel().copy()
        for tau in (10.0, 100.0, 1000.0, 1e4):
            def f(tt, tau=tau):
                T = tt.reshape(nf, NY); Y = rows @ T
                L = kl_rows(Q[pt], Y)
                val = 0.0; G = np.zeros_like(Y)
                for x in np.unique(pt):
                    sel = pt == x; Lx = L[sel]; mx = Lx.max()
                    e = np.exp(tau * (Lx - mx)); s = e.sum()
                    val += w[x] * (mx + np.log(s) / tau)
                    G[sel] = (w[x] * e / s)[:, None] * (softmax(Y[sel]) - Q[pt[sel]])
                return val, (rows.T @ G).ravel()
            r = minimize(f, t, jac=True, method="L-BFGS-B", options=dict(maxiter=5000, ftol=1e-15, gtol=1e-10))
            t = r.x
            best = min(best, worst_case(Q, rows, pt, w, t.reshape(nf, NY)))
    return best


def make_instance():
    n = 12
    inR = {k: np.isin(np.arange(n), v) for k, v in {1: range(0, 4), 2: range(4, 8), 3: range(8, 10)}.items()}
    p = rng.dirichlet(np.ones(n) * 3)
    z = {}
    for u in MEMB:
        zz = np.full(n, 0.0 if u == "u1" else -rng.uniform(0.5, 1.5))   # u1 sits at the threshold outside
        zz[inR[HOME[u]]] = rng.uniform(0.2, 2, inR[HOME[u]].sum())
        z[u] = zz
    h = {}
    for u in ("k1", "k2"):
        h[u] = inR[HOME[u]].astype(float)
    for u in ("e1", "e2", "f1", "f2"):
        hh = np.zeros(n); hh[inR[HOME[u]]] = rng.uniform(0, 2, inR[HOME[u]].sum()); h[u] = hh
    gv = np.zeros(n)
    for k in (1, 2, 3):
        gv[inR[k]] = rng.uniform(0.4, 1)
    gv[10:] = rng.uniform(0.4, 1, 2)
    h["g"] = gv
    W = {u: rng.normal(0, 1.2, NY) for u in MEMB + KEPT}
    cN = rng.normal(0, 1, NY)
    box = dict(m=rng.normal(0, 1, 3), beta=rng.normal(0, 0.5))
    # perturbation direction (clean table invariant)
    D = dict(h={u: np.zeros(n) for u in KEPT}, z={u: np.zeros(n) for u in MEMB})
    for u in ("k1", "e1", "k2", "e2", "f1", "f2"):
        out = np.where(~inR[HOME[u]])[0]
        pick = rng.choice(out, 3, replace=False)
        D["h"][u][pick] = rng.uniform(0.2, 1, 3)
    for u in MEMB:
        out = ~inR[HOME[u]]
        v = rng.normal(0, 1, out.sum()); v -= (p[out] @ v) / p[out].sum()   # zero p-mean outside the region
        D["z"][u][out] = v
    for k in (1, 2, 3):
        v = rng.normal(0, 0.5, inR[k].sum()); v -= (p[inR[k]] @ v) / p[inR[k]].sum()
        D["h"]["g"][inR[k]] = v
    for u in ("k1", "k2"):
        D["h"][u][inR[HOME[u]]] = rng.normal(0, 1, inR[HOME[u]].sum())       # judgment noise (no clean value)
    return dict(n=n, inR=inR, p=p, z=z, h=h, W=W, cN=cN, box=box, D=D)


def table(I, eps):
    z = {u: I["z"][u] + eps * I["D"]["z"][u] for u in MEMB}
    h = {u: I["h"][u] + eps * I["D"]["h"][u] for u in KEPT}
    for u in MEMB:
        h[u] = np.maximum(z[u], 0)
    return dict(z=z, h=h)


def clean_of(I, T):
    p = I["p"]; inR = I["inR"]
    z0 = {}; h0 = {}
    for u in MEMB:
        out = ~inR[HOME[u]]; zz = T["z"][u].copy(); zz[out] = (p[out] @ zz[out]) / p[out].sum(); z0[u] = zz
        hh = T["h"][u].copy(); hh[out] = 0.0; h0[u] = hh
    for u in KEPT:
        hh = T["h"][u].copy()
        if u == "g":
            for k in (1, 2, 3):
                hh[inR[k]] = (p[inR[k]] @ hh[inR[k]]) / p[inR[k]].sum()
        else:
            hh[~inR[HOME[u]]] = 0.0
            if u in ("k1", "k2"):
                # judgment claim (R, {k}, a = 1): clean value h + (1 - s) / (|K'| a_k), here 1 on R
                hh[inR[HOME[u]]] = 1.0
        h0[u] = hh
    return dict(z=z0, h=h0)


def boxv(I, z):
    m = I["box"]["m"]
    return np.maximum(m[0] * z["u3"] + m[1] * z["u4"] + m[2] * z["u5"] + I["box"]["beta"], 0)


def run(I, eps):
    n = I["n"]; p = I["p"]; inR = I["inR"]; W = I["W"]; cN = I["cN"]; m = I["box"]["m"]
    act = table(I, eps)
    cl = clean_of(I, act)
    base_clean = clean_of(I, table(I, 0.0))
    inv = max(max(np.abs(cl["h"][u] - base_clean["h"][u]).max() for u in KEPT),
              max(np.abs(cl["z"][u] - base_clean["z"][u]).max() for u in MEMB))
    Y_N = np.stack([sum(W[u] * act["h"][u][x] for u in MEMB + KEPT) + cN for x in range(n)])
    Q = softmax(Y_N)
    one = np.ones(n)
    L1, T1 = fit_head(Q, np.c_[cl["h"]["k1"], cl["h"]["e1"], one], p * inR[1])
    L2, T2 = fit_head(Q, np.c_[boxv(I, cl["z"]), cl["h"]["k2"], cl["h"]["e2"], one], p * inR[2])
    L0 = L1 + L2
    out2 = np.where(~inR[2])[0]
    vbox = boxv(I, {u: cl["z"][u][out2[:1]] for u in ("u3", "u4", "u5")})[0]     # box at the clean shelter value
    vS2 = T2[0] * vbox
    gR = {k: cl["h"]["g"][inR[k]][0] for k in (1, 2)}
    tR1 = T1[2] - cN - W["g"] * gR[1]
    tR2 = T2[3] - cN + vS2 - W["g"] * gR[2]
    W0 = dict(k1=T1[0] + tR1, e1=T1[1], k2=T2[1] + tR2, e2=T2[2], f1=W["f1"], f2=W["f2"], g=W["g"])
    V0 = T2[0]; cB = cN - vS2
    region = np.zeros(n, int)
    for k in (1, 2, 3):
        region[inR[k]] = k
    # E of Theorem TE4, by term
    Ea = Eb = Ec = Ed = 0.0
    for x in range(n):
        Ea += p[x] * sum(osc(W0[u]) * abs(act["h"][u][x] - cl["h"][u][x]) for u in KEPT)
        Eb += p[x] * osc(V0) * sum(abs(m[i]) * abs(act["z"][u][x] - cl["z"][u][x]) for i, u in enumerate(("u3", "u4", "u5")))
        if region[x] in (0, 3):
            Ed += p[x] * osc(sum(W[u] * (act["h"][u][x] - cl["h"][u][x]) for u in MEMB + KEPT))
    E = Ea + Eb + Ec + Ed
    # vertex rows of U (features: box, kept units, 1); B^0's head in the same features
    T_B0 = np.vstack([V0[None], np.stack([W0[u] for u in KEPT]), cB[None]])
    rows, pt = [], []
    for x in range(n):
        moved = [("h", u) for u in KEPT if act["h"][u][x] != cl["h"][u][x]] + \
                [("z", u) for u in ("u3", "u4", "u5") if act["z"][u][x] != cl["z"][u][x]]
        for bits in itertools.product((0, 1), repeat=len(moved)):
            hx = {u: cl["h"][u][x] for u in KEPT}; zx = {u: cl["z"][u][x] for u in ("u3", "u4", "u5")}
            for (kind, u), b in zip(moved, bits):
                if b:
                    (hx if kind == "h" else zx)[u] = (act["h"] if kind == "h" else act["z"])[u][x]
            bx = max(m[0] * zx["u3"] + m[1] * zx["u4"] + m[2] * zx["u5"] + I["box"]["beta"], 0)
            rows.append(np.r_[bx, [hx[u] for u in KEPT], 1.0]); pt.append(x)
    rows = np.array(rows); pt = np.array(pt)
    sup_B0 = worst_case(Q, rows, pt, p, T_B0)
    Fact = np.c_[boxv(I, act["z"]), np.stack([act["h"][u] for u in KEPT], 1), one]
    D_B0 = float(p @ kl_rows(Q, Fact @ T_B0))
    Vb, T_act = fit_head(Q, Fact, p, T_B0)
    Vrob = robust_min(Q, rows, pt, p, [T_B0, T_act]) if eps > 0 else D_B0
    return dict(eps=eps, clean_invariance=inv, L0=L0, V_rob_found=Vrob, sup_U_B0=sup_B0, D_B0_N=D_B0, E=E,
                Ea=Ea, Eb=Eb, Ec=Ec, Ed=Ed, bound=L0 + E, V_b_N=Vb, n_vertex_rows=int(len(rows)))


rows_out = []
ok = True
for inst in range(5):
    I = make_instance()
    r0 = run(I, 0.0)
    ident = abs(r0["D_B0_N"] - r0["L0"]) < 1e-9 and r0["E"] < 1e-12
    ok &= ident
    print(f"instance {inst}: eps=0: D(B0)={r0['D_B0_N']:.10f} sum L0={r0['L0']:.10f} E={r0['E']:.1e} ->",
          "TM1 identity HELD" if ident else "MISMATCH", flush=True)
    for eps in (0.1, 0.03, 0.01, 0.003, 0.001):
        r = run(I, eps)
        chain = (r["L0"] - 1e-8 <= r["V_rob_found"] <= r["sup_U_B0"] + 1e-10 <= r["bound"] + 2e-10
                 and r["D_B0_N"] <= r["sup_U_B0"] + 1e-12 and r["clean_invariance"] < 1e-12)
        ok &= chain
        r.update(inst=inst, chain=bool(chain)); rows_out.append(r)
        print(f"   eps={eps:<6} L0={r['L0']:.6f} <= robust found={r['V_rob_found']:.6f} <= sup_U D(B0)="
              f"{r['sup_U_B0']:.6f} <= L0+E={r['bound']:.6f} | E/eps={r['E'] / eps:8.3f} "
              f"(a {r['Ea'] / eps:.2f} b {r['Eb'] / eps:.2f} c {r['Ec'] / eps:.2f} d {r['Ed'] / eps:.2f}) | "
              f"L0 - V_b(N)={r['L0'] - r['V_b_N']:.5f} | {'ok' if chain else 'BROKEN'}", flush=True)
print("Theorem TE4 chain:", "HELD" if ok else "FAILED")
json.dump(rows_out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "te_robust_e.json"), "w"), indent=1)
