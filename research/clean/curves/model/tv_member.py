# serves: TV18
"""Exact checks of Part B.V.6 (COARSE-THEORY.md): Definition DM16, Theorem TV18, Corollary TV18.1.
Dense random pure-stream networks (d = 10, three blocks of 12 neurons, C = 4 logits), tanh and ReLU.
  A  TV18: for EVERY sum of members D, the cut's D_P <= the bound in the member claims' errors (norm form: orthogonal
     members, projection cut with constants eps + c(D); no-norm form: a random direct decomposition, oblique cut)
  B  the exact case: members that no reader reads (and, with norms, of constant energy) have claims 0, and dropping
     them loses nothing
  C  the knapsack dynamic programme of TV18.1 against all subsets
  D  the rule of TV18.1 on random zooms: the bound holds at every level, and nests
Run: python model/tv_member.py [instances]"""
import itertools
import sys
import numpy as np

rng = np.random.default_rng(1818)
N_INST = int(sys.argv[1]) if len(sys.argv) > 1 else 20
d, C, NX, L, q, DX = 10, 4, 64, 3, 12, 6
SIZES = [2, 2, 2, 2, 2]
Pperp = np.eye(C) - np.ones((C, C)) / C
fails, notes, COUNT = [], [], {}


def check(name, ok, info=""):
    COUNT[name.split()[0]] = COUNT.get(name.split()[0], 0) + 1
    if not ok:
        fails.append(f"{name} {info}")


ACT = {"tanh": (np.tanh, 1.0), "relu": (lambda a: np.maximum(a, 0), 1.0)}


def nrm(h, eps, on):
    return h / np.sqrt((h * h).sum(-1, keepdims=True) + eps) if on else h


def logsm(y):
    m = y.max(-1, keepdims=True)
    return y - m - np.log(np.exp(y - m).sum(-1, keepdims=True))


def kl(p, yN, yB):
    a, b = logsm(yN), logsm(yB)
    return float((p * (np.exp(a) * (a - b)).sum(-1)).sum())


def wn(p, f):
    return np.sqrt(float((p * f ** 2).sum()))


def make_net(norm, kind):
    net = dict(act=kind, eps=[0.5] * (L + 1), norm=norm)
    X = rng.normal(size=(NX, DX))
    net["e"] = np.tanh(X @ rng.normal(size=(DX, d))) * 2 + X[:, :1] * 0.5
    net["R"] = [rng.normal(size=(q, d)) / np.sqrt(d) * (3 if norm else 1) for _ in range(L)]
    net["b"] = [rng.normal(size=q) * .3 for _ in range(L)]
    net["W"] = [rng.normal(size=(d, q)) / np.sqrt(q) * .7 for _ in range(L)]
    net["c"] = [rng.normal(size=d) * .1 for _ in range(L)]
    net["U"] = rng.normal(size=(C, d)) * (2 if norm else .5)
    net["cy"] = rng.normal(size=C)
    return net


def forward_N(net):
    f = ACT[net["act"]][0]
    h = net["e"]
    hs = [h]
    for l in range(L):
        h = h + f(nrm(h, net["eps"][l], net["norm"]) @ net["R"][l].T + net["b"][l]) @ net["W"][l].T + net["c"][l]
        hs.append(h)
    return hs, nrm(h, net["eps"][L], net["norm"]) @ net["U"].T + net["cy"]


def cut(net, Q, epsc):
    """Projection cut (Q = Pi) or oblique cut (Q oblique): first write Q e, writes Q W, reads R Q, logits U Q."""
    f = ACT[net["act"]][0]
    h = net["e"] @ Q.T
    for l in range(L):
        h = h + (f(nrm(h, epsc[l], net["norm"]) @ (net["R"][l] @ Q).T + net["b"][l]) @ net["W"][l].T
                 + net["c"][l]) @ Q.T
    return nrm(h, epsc[L], net["norm"]) @ (net["U"] @ Q).T + net["cy"]


def members(orth):
    Bm = np.linalg.qr(rng.normal(size=(d, d)))[0]
    if not orth:   # at an angle, but not badly conditioned
        Bm = Bm + 0.4 * rng.normal(size=(d, d)) / np.sqrt(d)
    Bi = np.linalg.inv(Bm)
    idx = np.split(np.arange(d), np.cumsum(SIZES)[:-1])
    return [Bm[:, I] @ Bi[I] for I in idx]


def member_claims(net, QA, hs, p):
    """Definition DM16: eps_mc(B, l) for every member B and l = 1..L+1 (index L is the logits)."""
    norm, Lip = net["norm"], ACT[net["act"]][1]
    m = len(QA)
    nu = [np.sqrt((hs[l] ** 2).sum(1) + net["eps"][l]) if norm else np.ones(NX) for l in range(L + 1)]
    Ts = []
    for l in range(L):
        R, W = net["R"][l], net["W"][l]
        aw = np.array([[np.linalg.norm(QA[A] @ W[:, i]) for i in range(q)] for A in range(m)])
        br = np.array([[np.linalg.norm(QA[F].T @ R[i]) for i in range(q)] for F in range(m)])
        rn = np.linalg.norm(R, axis=1)
        Ts.append(Lip * aw @ ((br + rn[None]) / np.sqrt(net["eps"][l]) if norm else br).T)
    PU = Pperp @ net["U"]
    u = np.array([np.linalg.norm(PU @ QA[A], 2) for A in range(m)])
    if norm:
        u = (u + np.linalg.norm(PU, 2)) / np.sqrt(net["eps"][L])
    eps_mc = np.zeros((m, L + 1))
    for B in range(m):
        for l in range(L):
            QBh = hs[l] @ QA[B].T
            mu = np.abs(QBh @ net["R"][l].T) / nu[l][:, None]
            if norm:
                en = (QBh ** 2).sum(1)
                mu = mu + np.linalg.norm(net["R"][l], axis=1)[None] * np.abs(en - p @ en)[:, None] / nu[l][:, None] ** 2
            aw = np.array([[np.linalg.norm(QA[A] @ net["W"][l][:, i]) for i in range(q)] for A in range(m)])
            lam = Lip * mu @ aw.T                       # x by area
            Ml = np.eye(m)
            for k in range(l + 1, L):
                Ml = (np.eye(m) + Ts[k]) @ Ml
            eps_mc[B, l] = float(u @ Ml @ np.sqrt((p[:, None] * lam ** 2).sum(0))) ** 2
        QBh = hs[L] @ QA[B].T
        om = np.linalg.norm(QBh @ PU.T, axis=1) / nu[L]
        if norm:
            en = (QBh ** 2).sum(1)
            om = om + np.linalg.norm(PU, 2) * np.abs(en - p @ en) / nu[L] ** 2
        eps_mc[B, L] = wn(p, om) ** 2
    return eps_mc


def cut_for(net, QA, Dset, hs, p):
    Q = sum(QA[B] for B in range(len(QA)) if B not in Dset) if len(Dset) < len(QA) else np.zeros((d, d))
    if net["norm"]:
        cD = [float(p @ sum(((hs[l] @ QA[B].T) ** 2).sum(1) for B in Dset)) if Dset else 0.0 for l in range(L + 1)]
        epsc = [net["eps"][l] + cD[l] for l in range(L + 1)]
    else:
        epsc = net["eps"]
    return cut(net, Q, epsc)


ratios = []
for norm in (False, True):
    for kind in ACT:
        for inst in range(N_INST):
            net = make_net(norm, kind)
            QA = members(orth=norm)
            p = rng.dirichlet(np.ones(NX))
            hs, yN = forward_N(net)
            E = member_claims(net, QA, hs, p)
            s = np.sqrt(E).sum(1)
            for r in range(1, len(QA) + 1):
                for Dset in itertools.combinations(range(len(QA)), r):
                    yB = cut_for(net, QA, set(Dset), hs, p)
                    DP = kl(p, yN, yB)
                    bnd = 0.25 * s[list(Dset)].sum() ** 2
                    check(f"A {norm} {kind} bound", DP <= bnd * (1 + 1e-9) + 1e-12, f"{DP:.3e} > {bnd:.3e} D={Dset}")
                    ratios.append((norm, bnd / max(DP, 1e-300)))

# B: the exact case
for norm in (False, True):
    for kind in ACT:
        for inst in range(max(N_INST // 2, 3)):
            net = make_net(norm, kind)
            QA = members(orth=norm)
            dead = [0, 1]                                   # members no reader reads, never written
            Pd = sum(QA[B] for B in dead)
            Pk = np.eye(d) - Pd                             # projection onto the other members along the dead ones
            net["R"] = [R @ Pk for R in net["R"]]
            net["U"] = net["U"] @ Pk
            net["W"] = [Pk @ W for W in net["W"]]
            net["c"] = [Pk @ c for c in net["c"]]
            ek = net["e"] @ Pk.T
            if norm:                                        # constant energy in each dead member
                parts = []
                for B in dead:
                    v = rng.normal(size=(NX, d)) @ QA[B].T
                    parts.append(v / np.linalg.norm(v, axis=1, keepdims=True) * (1.0 + B))
                net["e"] = ek + sum(parts)
            else:
                net["e"] = ek + rng.normal(size=(NX, d)) @ Pd.T
            p = rng.dirichlet(np.ones(NX))
            hs, yN = forward_N(net)
            E = member_claims(net, QA, hs, p)
            live = [B for B in range(len(QA)) if B not in dead]
            check(f"B {norm} {kind} claims 0", np.abs(E[dead]).max() <= 1e-18 * max(1.0, np.abs(E[live]).max()),
                  f"{np.abs(E[dead]).max():.2e} vs {np.abs(E[live]).max():.2e}")
            yB = cut_for(net, QA, set(dead), hs, p)
            check(f"B {norm} {kind} loss 0", kl(p, yN, yB) < 1e-12, f"{kl(p, yN, yB):.2e}")


# C: the knapsack
def knap(cost, dims, need):
    """Least total cost of a subset with total dim >= need; returns (cost, subset)."""
    best = {0: (0.0, ())}
    for i, (c_, w_) in enumerate(zip(cost, dims)):
        new = dict(best)
        for t, (v, S) in best.items():
            t2 = min(t + w_, need)
            if t2 not in new or v + c_ < new[t2][0]:
                new[t2] = (v + c_, S + (i,))
        best = new
    return best.get(need, (np.inf, None))


for _ in range(300):
    k = rng.integers(1, 7)
    cost = rng.exponential(size=k)
    dims = rng.integers(1, 4, size=k)
    need = int(rng.integers(0, dims.sum() + 1))
    v, S = knap(cost, dims, need)
    brute = min((cost[list(T)].sum() for r in range(k + 1) for T in itertools.combinations(range(k), r)
                 if dims[list(T)].sum() >= need), default=np.inf)
    check("C knapsack", abs(v - brute) < 1e-12)

# D: the rule on random zooms (groups = sums of members; level j+1 merges groups of level j)
for norm in (False, True):
    for inst in range(max(N_INST // 2, 3)):
        net = make_net(norm, "tanh")
        QA = members(orth=norm)
        p = rng.dirichlet(np.ones(NX))
        hs, yN = forward_N(net)
        s = np.sqrt(member_claims(net, QA, hs, p)).sum(1)
        groups = [[0, 1], [2, 3], [4]]
        zoom = [groups, [[0, 1, 2, 3], [4]]]
        for lev in zoom:
            Dj = set()
            for S in lev:
                k_S = int(rng.integers(0, sum(SIZES[B] for B in S)))
                v, T = knap([s[B] for B in S], [SIZES[B] for B in S], sum(SIZES[B] for B in S) - k_S)
                Dj |= {S[t] for t in T}
            yB = cut_for(net, QA, Dj, hs, p)
            bnd = 0.25 * sum(s[B] for B in Dj) ** 2
            check(f"D {norm} rule bound", kl(p, yN, yB) <= bnd * (1 + 1e-9) + 1e-12)

# E: Proposition TV18.2 (norm form, members at an angle)
def claims_angle(net, QA, hs, p):
    Lip, m = ACT[net["act"]][1], len(QA)
    nu = [np.sqrt((hs[l] ** 2).sum(1) + net["eps"][l]) for l in range(L + 1)]
    G = [np.einsum("bxd,fxd->bfx", np.stack([hs[l] @ Q.T for Q in QA]), np.stack([hs[l] @ Q.T for Q in QA]))
         for l in range(L + 1)]
    gb = [np.einsum("bfx,x->bf", g, p) for g in G]
    cmin = [0.0] * (L + 1)                                      # Proposition TV18.2 as fixed: no floor, no premise
    gam = [np.abs(G[l][np.arange(m), np.arange(m)] - gb[l][np.arange(m), np.arange(m)][:, None])
           + 2 * np.array([sum(np.abs(G[l][B, F] - gb[l][B, F]) + abs(min(gb[l][B, F], 0)) for F in range(m) if F != B)
                           for B in range(m)])
           for l in range(L + 1)]                               # member by x
    sr = lambda r: sum(np.linalg.norm(Q.T @ r) for Q in QA)
    Ts = []
    for l in range(L):
        R, W = net["R"][l], net["W"][l]
        aw = np.array([[np.linalg.norm(QA[A] @ W[:, i]) for i in range(q)] for A in range(m)])
        br = np.array([[np.linalg.norm(QA[F].T @ R[i]) + sr(R[i]) for i in range(q)] for F in range(m)])
        Ts.append(Lip / np.sqrt(net["eps"][l] + cmin[l]) * aw @ br.T)
    PU = Pperp @ net["U"]
    sU = sum(np.linalg.norm(PU @ Q, 2) for Q in QA)
    u = np.array([np.linalg.norm(PU @ QA[A], 2) + sU for A in range(m)]) / np.sqrt(net["eps"][L] + cmin[L])
    E = np.zeros((m, L + 1))
    for B in range(m):
        for l in range(L):
            QBh = hs[l] @ QA[B].T
            mu = np.abs(QBh @ net["R"][l].T) / nu[l][:, None] + \
                np.linalg.norm(net["R"][l], axis=1)[None] * gam[l][B][:, None] / nu[l][:, None] ** 2
            aw = np.array([[np.linalg.norm(QA[A] @ net["W"][l][:, i]) for i in range(q)] for A in range(m)])
            lam = Lip * mu @ aw.T
            Ml = np.eye(m)
            for k in range(l + 1, L):
                Ml = (np.eye(m) + Ts[k]) @ Ml
            E[B, l] = float(u @ Ml @ np.sqrt((p[:, None] * lam ** 2).sum(0))) ** 2
        om = np.linalg.norm((hs[L] @ QA[B].T) @ PU.T, axis=1) / nu[L] + np.linalg.norm(PU, 2) * gam[L][B] / nu[L] ** 2
        E[B, L] = wn(p, om) ** 2
    return E, cmin


skipped, done = 0, 0
for kind in ACT:
    for inst in range(N_INST):
        net = make_net(True, kind)
        net["eps"] = [0.5] * (L + 1)
        Bm = np.linalg.qr(rng.normal(size=(d, d)))[0] + 0.15 * rng.normal(size=(d, d)) / np.sqrt(d)
        Bi = np.linalg.inv(Bm)
        QA = [Bm[:, I] @ Bi[I] for I in np.split(np.arange(d), np.cumsum(SIZES)[:-1])]
        p = rng.dirichlet(np.ones(NX))
        hs, yN = forward_N(net)
        E, cmin = claims_angle(net, QA, hs, p)
        if E is None:
            skipped += 1
            continue
        done += 1
        s = np.sqrt(E).sum(1)
        for r in range(1, len(QA) + 1):
            for Dset in itertools.combinations(range(len(QA)), r):
                Q = sum(QA[B] for B in range(len(QA)) if B not in Dset) if r < len(QA) else np.zeros((d, d))
                cD = [float(p @ ((hs[l] ** 2).sum(1) - ((hs[l] @ Q.T) ** 2).sum(1))) for l in range(L + 1)]
                yB = cut(net, Q, [net["eps"][l] + max(cD[l], 0.0) for l in range(L + 1)])
                DP = kl(p, yN, yB)
                bnd = 0.25 * s[list(Dset)].sum() ** 2
                check(f"E {kind} bound", DP <= bnd * (1 + 1e-9) + 1e-12, f"{DP:.3e} > {bnd:.3e}")
notes.append(f"E: {done} networks checked, {skipped} skipped (eps + c^- <= 0)")

# F: Corollary TV2.1, an exact instance at an angle
for kind in ACT:
    for inst in range(5):
        net = make_net(True, kind)
        Bm = np.linalg.qr(rng.normal(size=(d, d)))[0] + 0.3 * rng.normal(size=(d, d)) / np.sqrt(d)
        Bi = np.linalg.inv(Bm)
        idx = np.split(np.arange(d), np.cumsum(SIZES)[:-1])
        QA = [Bm[:, I] @ Bi[I] for I in idx]
        S_span = Bm[:, np.concatenate(idx[:4])]                  # S1 = B0 + B1, S2 = B2 + B3; R = B4
        Dcols, Kcols = np.concatenate([idx[1], idx[3]]), np.concatenate([idx[0], idx[2], idx[4]])
        Q = sum(QA[B] for B in (0, 2, 4))
        Ps = S_span @ np.linalg.pinv(S_span)
        net["R"] = [R @ (np.eye(d) - Ps) for R in net["R"]]     # no block reads S1 or S2
        net["U"] = net["U"] @ Q                                   # logits read nothing of D (along K)
        v = Bm[:, Dcols] @ rng.normal(size=len(Dcols)); v /= np.linalg.norm(v)
        Kb = np.linalg.qr(Bm[:, Kcols])[0]
        kv = Kb.T @ (Q.T @ v)                                     # writes inside K and orthogonal to Q^T v
        Pw_basis = np.linalg.qr(Kb - np.outer(Kb @ (Kb.T @ (Q.T @ v)), Kb.T @ (Q.T @ v)) / max(kv @ kv, 1e-12))[0][:, :len(Kcols) - 1]
        Pw = Pw_basis @ Pw_basis.T
        net["W"] = [Pw @ W for W in net["W"]]
        net["c"] = [Pw @ c for c in net["c"]]
        k0 = net["e"] @ Q.T
        a_ = k0 @ v
        cc_ = 0.8
        sx = -a_ + np.sqrt(a_ ** 2 + cc_)                         # |k0 + s v|^2 - |k0|^2 = cc_
        net["e"] = k0 + sx[:, None] * v[None]
        p = rng.dirichlet(np.ones(NX))
        hs, yN = forward_N(net)
        eq = [(hs[l] ** 2).sum(1) - ((hs[l] @ Q.T) ** 2).sum(1) for l in range(L + 1)]
        check(f"F {kind} oblique energy constant", all(np.ptp(e_) < 1e-9 for e_ in eq))
        yB = cut(net, Q, [net["eps"][l] + float(p @ eq[l]) for l in range(L + 1)])
        check(f"F {kind} exact", kl(p, yN, yB) < 1e-12, f"{kl(p, yN, yB):.2e}")

# G: the comment after TV18 (referee BV6 finding 2): an exact cut whose member claims are > 0
for t_ in (0.3, 0.9):
    netg = dict(act="tanh", norm=True, eps=[0.5, 0.5], e=None)
    tx = np.array([t_, -t_ + 1.0] * 32)
    e = np.stack([np.cos(tx), np.sin(tx), np.ones_like(tx)], 1)
    Lg = 1
    Rg, Wg, Ug = np.array([[0, 0, 1.0]]), np.array([[0], [0], [1.0]]), np.array([[0, 0, 2.0], [0, 0, 0]])
    nu0 = np.sqrt((e ** 2).sum(1) + 0.5)
    h1 = e + np.tanh(e @ Rg.T / nu0[:, None]) @ Wg.T
    PD = np.diag([1.0, 1, 0])
    yN = (h1 / np.sqrt((h1 ** 2).sum(1, keepdims=True) + 0.5)) @ Ug.T
    c0, c1 = float(np.mean((e @ PD) ** 2 @ np.ones(3))), float(np.mean((h1 @ PD) ** 2 @ np.ones(3)))
    Pi = np.eye(3) - PD
    hc0 = e @ Pi
    hc1 = hc0 + np.tanh(hc0 @ Rg.T / np.sqrt((hc0 ** 2).sum(1) + 0.5 + c0)[:, None]) @ Wg.T
    yB = (hc1 / np.sqrt((hc1 ** 2).sum(1, keepdims=True) + 0.5 + c1)) @ Ug.T
    pw = np.full(len(tx), 1 / len(tx))
    check("G exact cut", kl(pw, yN, yB) < 1e-14, f"{kl(pw, yN, yB):.2e}")
    en = (e[:, 0]) ** 2
    mu = abs(Rg[0] @ np.array([1.0, 0, 0])) + np.linalg.norm(Rg[0]) * np.abs(en - en.mean()) / nu0 ** 2
    check("G member claim > 0", wn(pw, mu) > 1e-3)

for norm in (False, True):
    r = [x for n_, x in ratios if n_ == norm]
    notes.append(f"A bound/D_P norm={norm}: median {np.median(r):.3g} (min {np.min(r):.3g})")
print("\n".join(notes))
print("checks run:", COUNT)
print(f"tv_member: {len(fails)} failures")
for f_ in fails[:20]:
    print("FAIL", f_)
