# serves: TV6
"""Exact checks of Part B.V.2 (COARSE-THEORY.md): Theorem TV6, Proposition TV7, Corollary TV6.1, Proposition TV8.

Instance: a dense random pure-stream network (Definition DV1), d = 10, three blocks of 12 neurons (so the readout
kernel K of Lemma TV5 is {0}: every direction is read), C = 4 logits, tanh or relu (Lipschitz 1), with norms
(eps_l = 0.5) or in the no-norm form. Areas: a random orthonormal basis cut into 5 areas of 2 directions. D = one or
two areas. Checks, per instance:
  A  TV6 part 1, pointwise: |P_perp (y_B - y_N)(x)| <= |omega(x)| + sum_l u^T M_l lambda_l(x) at every x, and
     D_P(B) <= the bound; for the projection cut (reads Pi r, logit matrix U Pi) and for a re-fit cut (reads, biases and
     logits re-fit by least squares)
  B  TV6 part 2 (no norm): leaks that flow into a dead-end area (read by no later neuron, not by the logits) add
     nothing: the bound equals (1/4) ||omega||^2 and D_P <= it
  C  TV7: the normed one-block family with the gain in the logits; every claim of the map (write claims, both forms of
     the block-read error for every area, readout and energy claims) <= delta^2, while the least loss of cuts that see
     x1 only stays c(s); also N's logits are (s x2, 0)
  D  TV6.1: the carried leak errors of DM14 (written from N, P and D alone) give the same bound as TV6
  E  TV8: for eta where the computed premise holds, D_P <= the TV8 bound, pointwise too
Usage: python tv_graded.py [n_instances]
"""
import sys
import numpy as np

rng = np.random.default_rng(31)
N_INST = int(sys.argv[1]) if len(sys.argv) > 1 else 40
d, C, NX, DX, L, q = 10, 4, 250, 6, 3, 12
Pperp = np.eye(C) - np.ones((C, C)) / C
fails = []


def check(name, ok, info=""):
    if not ok:
        fails.append(f"{name} {info}")


def logsm(y):
    m = y.max(-1, keepdims=True)
    return y - m - np.log(np.exp(y - m).sum(-1, keepdims=True))


def kl(p, yN, yB):
    a, b = logsm(yN), logsm(yB)
    return float((p * (np.exp(a) * (a - b)).sum(-1)).sum())


def nrm(h, eps, on):
    return h / np.sqrt((h * h).sum(-1, keepdims=True) + eps) if on else h


def proj(M):
    u, sv, _ = np.linalg.svd(M, full_matrices=False)
    u = u[:, sv > 1e-9 * max(1.0, sv.max() if sv.size else 1.0)]
    return u @ u.T


def act(t, kind):
    return np.tanh(t) if kind == "tanh" else np.maximum(t, 0)


def forward_N(net, norm):
    h = net["e"]
    hs, ns = [h], []
    for l in range(L):
        z = nrm(h, net["eps"][l], norm)
        ns.append(z)
        h = h + act(z @ net["R"][l].T + net["b"][l], net["act"]) @ net["W"][l].T + net["c"][l]
        hs.append(h)
    zL = nrm(h, net["eps"][L], norm)
    return hs, ns, zL, zL @ net["U"].T + net["cy"]


def forward_cut(net, cut, Pi, norm):
    h = net["e"] @ Pi.T
    for l in range(L):
        z = nrm(h, cut["eps"][l], norm)
        g = act(z @ cut["R"][l].T + cut["b"][l], net["act"])
        h = h + (g @ net["W"][l].T + net["c"][l]) @ Pi.T
    return nrm(h, cut["eps"][L], norm) @ cut["U"].T + net["cy"]


def tv6_terms(net, cut, Pi, areas, p, norm, eta=None):
    """Quantities of Theorem TV6; returns pointwise right side, the L2 bound, and the carried terms."""
    hs, ns, zL, yN = forward_N(net, norm)
    Lip = 1.0
    lam_pts, Ms, Ts, lam_norms, rhos = [], [], [], [], []
    for l in range(L):
        Ph = hs[l] @ Pi.T
        nup = np.sqrt((Ph ** 2).sum(-1) + cut["eps"][l]) if norm else np.ones(NX)
        rho = nup.min()
        zc = nrm(Ph, cut["eps"][l], norm)
        leak = zc @ cut["R"][l].T + cut["b"][l] - (ns[l] @ net["R"][l].T + net["b"][l])     # NX x q
        WA = np.array([[np.linalg.norm(QA.T @ (Pi @ net["W"][l][:, i])) for i in range(q)] for QA in areas])
        RB = np.array([[np.linalg.norm(QB.T @ cut["R"][l][i]) for i in range(q)] for QB in areas])
        rn = np.linalg.norm(cut["R"][l], axis=1)
        if eta is None:
            coup = rn
        else:   # Proposition TV8: largest read on P plus 2 eta |r'|
            coup = np.abs(zc @ cut["R"][l].T).max(0) + 2 * eta * rn
        T = Lip / rho * WA @ (RB + (coup[None, :] if norm else 0)).T
        rhos.append(rho)
        lam = Lip * np.abs(leak) @ WA.T                                                     # NX x m
        Ts.append(T)
        lam_pts.append(lam)
    m = len(areas)
    for l in range(L):
        M = np.eye(m)
        for k in range(l + 1, L):
            M = (np.eye(m) + Ts[k]) @ M
        Ms.append(M)
    PhL = hs[L] @ Pi.T
    nupL = np.sqrt((PhL ** 2).sum(-1) + cut["eps"][L]) if norm else np.ones(NX)
    rhoL = nupL.min()
    PU = Pperp @ cut["U"]
    zLc = nrm(PhL, cut["eps"][L], norm)
    cu = np.linalg.norm(PU, 2) if eta is None else np.linalg.norm(zLc @ PU.T, axis=1).max() + 2 * eta * np.linalg.norm(PU, 2)
    u = np.array([np.linalg.norm(PU @ QA, 2) + (cu if norm else 0) for QA in areas]) / rhoL
    rhos.append(rhoL)
    omega = (nrm(PhL, cut["eps"][L], norm) @ cut["U"].T - zL @ net["U"].T) @ Pperp.T
    right_pts = np.linalg.norm(omega, axis=1) + sum(lam_pts[l] @ (u @ Ms[l]) for l in range(L))
    om_n = np.sqrt(float((p * (omega ** 2).sum(-1)).sum()))
    carried = [float(u @ Ms[l] @ np.sqrt((p[:, None] * lam_pts[l] ** 2).sum(0))) for l in range(L)]
    bound = 0.25 * (om_n + sum(carried)) ** 2
    if eta is not None:   # the premise of TV8, computed pointwise
        ehat = np.zeros((NX, m))
        ok = True
        for l in range(L):
            ehat = ehat @ (np.eye(m) + Ts[l]).T + lam_pts[l]
            ok = ok and bool((ehat.sum(1) <= eta * rhos[l + 1]).all())
        return right_pts, bound, om_n, carried, yN, ok
    return right_pts, bound, om_n, carried, yN


def make_net(Q, norm, kind, dead=None, Dm=None):
    net = dict(act=kind, eps=[0.5] * (L + 1))
    X = rng.normal(size=(NX, DX))
    net["e"] = np.tanh(X @ rng.normal(size=(DX, d))) * 2 + X[:, :1] * 0.5
    net["R"] = [rng.normal(size=(q, d)) / np.sqrt(d) * (3 if norm else 1) for _ in range(L)]
    net["b"] = [rng.normal(size=q) * .3 for _ in range(L)]
    net["W"] = [rng.normal(size=(d, q)) / np.sqrt(q) * .7 for _ in range(L)]
    net["c"] = [rng.normal(size=d) * .1 for _ in range(L)]
    net["U"] = rng.normal(size=(C, d)) * (2 if norm else .5)
    net["cy"] = rng.normal(size=C)
    if dead is not None:  # part 2: neurons 0-3 of each block read D and write only into the dead area; nobody reads it
        Pz, PD = proj(dead), proj(Dm)
        for l in range(L):
            for i in range(q):
                net["R"][l][i] = net["R"][l][i] - Pz @ net["R"][l][i]
                if i < 4:
                    net["W"][l][:, i] = Pz @ net["W"][l][:, i] * 3
                else:
                    net["R"][l][i] = net["R"][l][i] - PD @ net["R"][l][i]
                    net["W"][l][:, i] = net["W"][l][:, i] - Pz @ net["W"][l][:, i]
            net["c"][l] = net["c"][l] - Pz @ net["c"][l]
        net["U"] = net["U"] - net["U"] @ Pz
        net["e"] = net["e"] - net["e"] @ Pz
    return net


def refit(net, Pi, p, norm, hs, ns, zL, base):
    """Re-fit reads, biases and logits by weighted least squares on the kept directions (one admissible cut)."""
    cut = dict(base)
    Bk = np.linalg.svd(Pi)[0][:, :int(round(np.trace(Pi)))]
    sw = np.sqrt(p)[:, None]
    Rn, bn = [], []
    for l in range(L):
        Z = nrm(hs[l] @ Pi.T, cut["eps"][l], norm) @ Bk
        F = np.concatenate([Z, np.ones((NX, 1))], 1)
        tgt = ns[l] @ net["R"][l].T + net["b"][l]
        coef, *_ = np.linalg.lstsq(sw * F, sw * tgt, rcond=None)
        Rn.append((Bk @ coef[:-1]).T)
        bn.append(coef[-1])
    cut["R"], cut["b"] = Rn, bn
    Z = nrm(hs[L] @ Pi.T, cut["eps"][L], norm) @ Bk
    tgt = zL @ net["U"].T @ Pperp.T
    coef, *_ = np.linalg.lstsq(sw * Z, sw * tgt, rcond=None)
    cut["U"] = (Bk @ coef).T
    return cut


ratios = []
tv8_gain = []
for inst in range(N_INST):
    norm = inst % 2 == 0
    kind = "tanh" if inst % 4 < 2 else "relu"
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    areas = [Q[:, 2 * j:2 * j + 2] for j in range(5)]
    p = rng.dirichlet(np.ones(NX))
    net = make_net(Q, norm, kind)
    check("K-generic", np.linalg.matrix_rank(np.concatenate(net["R"], 0)) == d)
    Dm = np.concatenate(areas[3:3 + 1 + inst % 2], 1)
    Pi = np.eye(d) - proj(Dm)
    hs, ns, zL, yN = forward_N(net, norm)
    cvals = [float((p * ((hh @ proj(Dm)) ** 2).sum(-1)).sum()) for hh in hs]
    eps_c = [net["eps"][l] + cvals[l] for l in range(L + 1)]
    proj_cut = dict(R=[r @ Pi for r in net["R"]], b=list(net["b"]), U=net["U"] @ Pi, eps=eps_c)
    for name, cut in (("proj", proj_cut), ("refit", refit(net, Pi, p, norm, hs, ns, zL, proj_cut))):
        yB = forward_cut(net, cut, Pi, norm)
        right, bound, om, carried, _ = tv6_terms(net, cut, Pi, areas, p, norm)
        left = np.linalg.norm((yB - yN) @ Pperp.T, axis=1)
        check(f"A-TV6-pointwise-{name}", (left <= right * (1 + 1e-9) + 1e-12).all(), (left - right).max())
        dp = kl(p, yN, yB)
        check(f"A-TV6-bound-{name}", dp <= bound * (1 + 1e-9) + 1e-14, (dp, bound))
        ratios.append(bound / max(dp, 1e-300))
        if name == "proj":  # D. Corollary TV6.1: the carried leak errors give the same bound
            eps_cl = [c_ ** 2 for c_ in carried] + [om ** 2]
            b61 = 0.25 * sum(np.sqrt(e_) for e_ in eps_cl) ** 2
            check("D-TV6.1", abs(b61 - bound) <= 1e-9 * max(1, bound) and dp <= b61 * (1 + 1e-9) + 1e-14)

    # E. Proposition TV8 (norm form, projection cut), on a copy of N whose blocks read D only weakly (kappa = 0.01)
    #    and whose stream holds little in D, so the leaks are small
    if norm:
        PDm = proj(Dm)
        net = dict(net, R=[r - 0.99 * r @ PDm for r in net["R"]], e=net["e"] - 0.99 * net["e"] @ PDm,
                   W=[w - 0.99 * PDm @ w for w in net["W"]], c=[c_ - 0.99 * PDm @ c_ for c_ in net["c"]])
        hs, ns, zL, yN = forward_N(net, norm)
        cvals = [float((p * ((hh @ PDm) ** 2).sum(-1)).sum()) for hh in hs]
        proj_cut = dict(R=[r @ Pi for r in net["R"]], b=list(net["b"]), U=net["U"] @ Pi,
                        eps=[net["eps"][l] + cvals[l] for l in range(L + 1)])
        yB = forward_cut(net, proj_cut, Pi, norm)
        right6, b6, _, _, _ = tv6_terms(net, proj_cut, Pi, areas, p, norm)
        left = np.linalg.norm((yB - yN) @ Pperp.T, axis=1)
        check("E-TV6-weak-pointwise", (left <= right6 * (1 + 1e-9) + 1e-12).all())
        dp = kl(p, yN, yB)
        left = np.linalg.norm((yB - yN) @ Pperp.T, axis=1)
        best = None
        for eta in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0):
            right8, b8, _, _, _, ok = tv6_terms(net, proj_cut, Pi, areas, p, norm, eta=eta)
            if ok:
                check("E-TV8-pointwise", (left <= right8 * (1 + 1e-9) + 1e-12).all())
                check("E-TV8-bound", dp <= b8 * (1 + 1e-9) + 1e-14, (eta, dp, b8))
                best = b8 if best is None else min(best, b8)
        tv8_gain.append(None if best is None else (b6 / best, b6 / max(dp, 1e-300), best / max(dp, 1e-300)))

    # B. no-norm routing: leaks into a dead-end area
    netb = make_net(Q, False, kind, dead=areas[4], Dm=areas[3])
    Pi = np.eye(d) - proj(areas[3])
    hs, ns, zL, yNb = forward_N(netb, False)
    cutb = dict(R=[r @ Pi for r in netb["R"]], b=list(netb["b"]), U=netb["U"] @ Pi, eps=netb["eps"])
    right, bound, om, carried, _ = tv6_terms(netb, cutb, Pi, areas, p, False)
    leak_present = any(np.abs(hs[l] @ proj(areas[3]) @ netb["R"][l][:4].T).max() > 1e-6 for l in range(L))
    yB = forward_cut(netb, cutb, Pi, False)
    check("B-TV6.2-deadend", leak_present and max(carried) < 1e-12 and abs(bound - om ** 2 / 4) < 1e-12
          and kl(p, yNb, yB) <= bound + 1e-14, (carried, bound, om ** 2 / 4, kl(p, yNb, yB)))

# C. Proposition TV7: normed (eps_l = 1), one block of one neuron, sigma = identity, gain in the logits
s_ = 1.5
qp, qm = np.exp(logsm(np.array([s_, 0.0]))), np.exp(logsm(np.array([-s_, 0.0])))
mix = (qp + qm) / 2
c_s = 0.5 * (float((qp * np.log(qp / mix)).sum()) + float((qm * np.log(qm / mix)).sum()))
P4 = np.array([[a, b] for a in (-1, 1) for b in (-1, 1)], float)
w4 = np.ones(4) / 4
rows = []
E3 = np.eye(3)
PA = {"e1": np.outer(E3[0], E3[0]), "D": np.outer(E3[1], E3[1]), "e12": np.outer(E3[0], E3[0]) + np.outer(E3[1], E3[1]),
      "e3": np.outer(E3[2], E3[2])}
for dl in (1.0, 1e-1, 1e-2, 1e-3, 1e-4):
    h0 = P4[:, :1] * E3[0] + P4[:, 1:] * E3[1]
    n0 = nrm(h0, 1.0, True)
    r1, w1 = dl * E3[1], E3[2]
    h1 = h0 + (n0 @ r1)[:, None] * w1
    nu0, nu1 = np.sqrt(3.0), np.sqrt((h1 ** 2).sum(-1) + 1)
    check("C-TV7-const-norm", np.ptp(nu1) < 1e-12)
    t = s_ * nu0 * nu1[0] / dl
    U = np.stack([t * E3[2], np.zeros(3)])
    yN = nrm(h1, 1.0, True) @ U.T
    errs = {"write e": float((w4 * ((h0 @ (np.eye(3) - PA["e12"])) ** 2).sum(-1)).sum()),
            "write 1": float(np.linalg.norm(w1 - PA["e3"] @ w1) ** 2),
            "readout": float(np.linalg.norm(U @ PA["D"]) ** 2),
            "energy": max(float(np.var(((hh @ PA["D"]) ** 2).sum(-1))) for hh in (h0, h1))}
    En0, Eh1 = float((w4 * (n0 ** 2).sum(-1)).sum()), float((w4 * (h1 ** 2).sum(-1)).sum())
    br = {a: float(np.linalg.norm(P @ r1) ** 2) for a, P in PA.items()}
    brw = {a: (np.linalg.norm(w1) ** 2 / Eh1) * br[a] * En0 for a in PA}
    allerr = list(errs.values()) + list(br.values()) + list(brw.values())
    best = np.inf
    for tt in np.linspace(-3, 3, 601):
        best = min(best, kl(w4, yN, np.stack([np.full(4, tt), np.zeros(4)], 1)))
    rows.append((dl, max(allerr), best))
    check("C-TV7-logits", np.abs(yN - np.stack([s_ * P4[:, 1], np.zeros(4)], 1)).max() < 1e-9)
    check("C-TV7-claims", max(errs.values()) < 1e-20 and max(allerr) <= dl ** 2 * (1 + 1e-12), (errs, br, brw))
    check("C-TV7-brw", abs(brw["D"] - (2 / 3) * dl ** 2 / (2 + dl ** 2 / 3)) < 1e-14)
    check("C-TV7-loss", best >= c_s - 1e-9 and abs(best - c_s) < 1e-4, (dl, best, c_s))

print(f"instances {N_INST}; failures {len(fails)}")
for f in fails[:20]:
    print("  FAIL", f)
r = np.array(ratios)
print(f"TV6 bound / D_P (dense networks, K = 0): median {np.median(r):.3g}, range {r.min():.3g} to {r.max():.3g}")
print(f"TV7 (s = {s_}, c(s) = {c_s:.4f}): " + "; ".join(f"delta {e:g}: largest claim error {b:.1e}, least loss {v:.4f}" for e, b, v in rows))
g8 = [x for x in tv8_gain if x is not None]
if g8:
    g8 = np.array(g8)
    print(f"TV8 (weak leaks): premise held for some eta in {len(g8)} of {len(tv8_gain)} normed instances; "
          f"median TV6 bound / TV8 bound {np.median(g8[:, 0]):.3g}; median bound / D_P: TV6 {np.median(g8[:, 1]):.3g}, "
          f"TV8 {np.median(g8[:, 2]):.3g}")
else:
    print(f"TV8: premise held in 0 of {len(tv8_gain)} normed instances")
