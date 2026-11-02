# serves: TV9
"""Exact checks of Part B.V.3 (COARSE-THEORY.md): Fact F14, Theorem TV9, Propositions TV9.1 and TV9.2, Lemmas TV10 to
TV13. Reuses the networks and forward passes of tv_graded.py (dense random pure-stream networks, d = 10, three blocks
of 12 neurons, C = 4 logits, tanh or relu).
  A  F14: no norm, a random invertible A gives N's logits; norm, a random A changes them and each read is s_l times
     N's read; an orthogonal A gives N's logits
  B  TV10: an oblique cut along a random complement K computes the same logits as the re-fit cut it is built from
  C  TV11: part 1 builds r' (0 on D, r on E) when D and E meet only in 0; part 2 fails when they share a vector
  D  TV12: a cut with constants eps + c equals the cut with constants eps and one more constant, unread direction
  E  TV9: the closed-form read has the least leak (against the projection read and random reads in D perp) and the
     least value equals Var(t) - c^T Z^+ c; the logit fit likewise
  F  TV9.1 and TV9.2: the copy network (projection cut loss log 2 - H, B* exact); TV9.2's bound on TV7's network
     equals c(s) and is below every cut's loss; and below the projection and B* cuts on the dense networks
  G  TV13: the signed identity holds to rounding error at every point (no norm), for the projection cut and B*
Also prints: the TV6 bound over D_P for the projection cut and for B* separately, and where TV6 loses against the
exact signed sum (signs within a block, signs across blocks, slopes).
Usage: python tv_cancel.py [n_instances]
"""
import sys
import numpy as np

_argv = sys.argv
sys.argv = [_argv[0], "1"]   # tv_graded runs its own checks on import; 1 instance keeps that short
import tv_graded as G  # noqa: E402
sys.argv = _argv

rng = np.random.default_rng(57)
N_INST = int(sys.argv[1]) if len(sys.argv) > 1 else 40
d, C, NX, L, q = G.d, G.C, G.NX, G.L, G.q
Pperp = G.Pperp
fails = []


def check(name, ok, info=""):
    if not ok:
        fails.append(f"{name} {info}")


def fwd_general(net, norm, A=None):
    """N^A of Fact F14 (A = None gives N); returns the reads of each block and the logits."""
    Ai = None if A is None else np.linalg.inv(A)
    h = net["e"] if A is None else net["e"] @ A.T
    reads = []
    for l in range(L):
        z = G.nrm(h, net["eps"][l], norm)
        Rl = net["R"][l] if A is None else net["R"][l] @ Ai
        a = z @ Rl.T
        reads.append(a)
        W = net["W"][l] if A is None else A @ net["W"][l]
        c = net["c"][l] if A is None else A @ net["c"][l]
        h = h + G.act(a + net["b"][l], net["act"]) @ W.T + c
    U = net["U"] if A is None else net["U"] @ Ai
    return reads, G.nrm(h, net["eps"][L], norm) @ U.T + net["cy"], h


def cut_run(net, Rc, bc, Uc, cyc, epsc, Pi, norm, first=None, battery=None):
    """Re-fit cut in R^d coordinates (Lemma TV0) with projection Pi (orthogonal or oblique); returns stream list, logits.
    battery: list of c_l; adds one coordinate holding c_l^(1/2) before norm l (Lemma TV12)."""
    h = (net["e"] if first is None else first) @ Pi.T
    if battery is not None:
        h = np.concatenate([h, np.full((NX, 1), np.sqrt(battery[0]))], 1)
    hs = [h]
    for l in range(L):
        z = G.nrm(h, epsc[l], norm)
        Rl = Rc[l] if battery is None else np.concatenate([Rc[l], np.zeros((q, 1))], 1)
        g = G.act(z @ Rl.T + bc[l], net["act"])
        add = (g @ net["W"][l].T + net["c"][l]) @ Pi.T
        if battery is not None:
            add = np.concatenate([add, np.full((NX, 1), np.sqrt(battery[l + 1]) - np.sqrt(battery[l]))], 1)
        h = h + add
        hs.append(h)
    Uf = Uc if battery is None else np.concatenate([Uc, np.zeros((C, 1))], 1)
    return hs, G.nrm(h, epsc[L], norm) @ Uf.T + cyc


def tv9_cut(net, Pi, p, norm, hs, ns, zL, epsc):
    """B* of Theorem TV9: closed-form reads r* = Z^+ c, b* = Et - r* . zbar; logits by the same fit with intercept."""
    Rn, bn, best = [], [], []
    for l in range(L):
        z = G.nrm(hs[l] @ Pi.T, epsc[l], norm)
        t = ns[l] @ net["R"][l].T + net["b"][l]
        zb, tb = p @ z, p @ t
        Zc, tc = z - zb, t - tb
        Z = (Zc * p[:, None]).T @ Zc
        cc = (Zc * p[:, None]).T @ tc
        r = (np.linalg.pinv(Z, rcond=1e-12) @ cc).T
        Rn.append(r)
        bn.append(tb - r @ zb)
        best.append(p @ tc ** 2 - np.einsum("iq,ij,jq->q", cc, np.linalg.pinv(Z, rcond=1e-12), cc))
    z = G.nrm(hs[L] @ Pi.T, epsc[L], norm)
    t = (zL @ net["U"].T + net["cy"]) @ Pperp.T
    zb, tb = p @ z, p @ t
    Zc, tc = z - zb, t - tb
    Z = (Zc * p[:, None]).T @ Zc
    U = ((np.linalg.pinv(Z, rcond=1e-12) @ ((Zc * p[:, None]).T @ tc))).T
    return Rn, bn, U, tb - U @ zb, best


def leaks(net, Rc, bc, Pi, norm, hs, ns, epsc):
    out = []
    for l in range(L):
        z = G.nrm(hs[l] @ Pi.T, epsc[l], norm)
        out.append(z @ Rc[l].T + bc[l] - (ns[l] @ net["R"][l].T + net["b"][l]))
    return out


def tv6_bound(net, Rc, bc, Uc, cyc, epsc, Pi, areas, p, norm):
    """Theorem TV6's L2 bound (logit bias cyc allowed, TV9 part 3)."""
    cut = dict(R=Rc, b=bc, U=Uc, eps=epsc)
    hs, ns, zL, yN = G.forward_N(net, norm)
    right, bound, om, carried, _ = G.tv6_terms(net, cut, Pi, areas, p, norm)
    # tv6_terms uses N's bias c_y in omega; recompute omega with cyc
    PhL = hs[L] @ Pi.T
    omega = (G.nrm(PhL, epsc[L], norm) @ Uc.T + cyc - zL @ net["U"].T - net["cy"]) @ Pperp.T
    om2 = np.sqrt(float((p * (omega ** 2).sum(-1)).sum()))
    return 0.25 * (om2 + sum(carried)) ** 2


s_ = 1.5
qp, qm = np.exp(G.logsm(np.array([s_, 0.0]))), np.exp(G.logsm(np.array([-s_, 0.0])))
mix = (qp + qm) / 2
c_s = 0.5 * (float((qp * np.log(qp / mix)).sum()) + float((qm * np.log(qm / mix)).sum()))


def tv92_bound(p, yN, key):
    """Proposition TV9.2: E_P KL(q || qbar(key)), key = rounded Pi e(x)."""
    qN = np.exp(G.logsm(yN))
    _, inv = np.unique(np.round(key, 10), axis=0, return_inverse=True)
    inv = inv.ravel()
    tot = 0.0
    for g in np.unique(inv):
        m = inv == g
        qb = (p[m, None] * qN[m]).sum(0) / p[m].sum()
        tot += float((p[m] * (qN[m] * (np.log(qN[m]) - np.log(qb))).sum(-1)).sum())
    return tot


LIPP = 4 / (3 * np.sqrt(3))   # the Lipschitz constant of tanh' (max |tanh''|)


def tv14(net, Rc, bc, Uc, cyc, Pi, p, hs, yN, yB):
    """Proposition TV14 (no norm, tanh, Lip = 1): Lambda_N, rem pointwise, and D_P(B)."""
    lk = leaks(net, Rc, bc, Pi, False, hs, hs, [0] * (L + 1))   # no norm: the normed stream is the stream
    sN = [1 - np.tanh(hs[l] @ net["R"][l].T + net["b"][l]) ** 2 for l in range(L)]
    PW = [Pi @ net["W"][l] for l in range(L)]
    Lam = ((hs[L] @ Pi.T) @ Uc.T + cyc - hs[L] @ net["U"].T - net["cy"]) @ Pperp.T
    for l in range(L):
        v = (lk[l] * sN[l]) @ PW[l].T
        for m in range(l + 1, L):
            v = v + ((v @ Rc[m].T) * sN[m]) @ PW[m].T
        Lam = Lam + v @ Uc.T @ Pperp.T
    beta = np.zeros(NX)
    gam = np.zeros(NX)
    rn = [np.linalg.norm(Rc[l], axis=1) for l in range(L)]
    for l in range(L):
        nW, nR = np.linalg.norm(PW[l], 2), np.linalg.norm(Rc[l], 2)
        IJ = np.eye(d)[None] + np.einsum("dq,xq,qe->xde", PW[l], sN[l], Rc[l])
        nIJ = np.linalg.norm(IJ, ord=2, axis=(1, 2))
        gam = nIJ * gam + LIPP / 2 * nW * ((rn[l][None, :] * beta[:, None] + np.abs(lk[l])) ** 2).sum(1)
        beta = (1 + nW * nR) * beta + nW * np.linalg.norm(lk[l], axis=1)
    rem = np.linalg.norm(Pperp @ Uc, 2) * gam
    return Lam, rem, G.kl(p, yN, yB)


tv14_ratios = []
scale_rows = []
ratios = {"proj": [], "B*": []}
loss_split = []
gap_parts = []
for inst in range(N_INST):
    norm = inst % 2 == 0
    kind = "tanh" if inst % 4 < 2 else "relu"
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    areas = [Q[:, 2 * j:2 * j + 2] for j in range(5)]
    p = rng.dirichlet(np.ones(NX))
    net = G.make_net(Q, norm, kind)
    Dm = np.concatenate(areas[3:3 + 1 + inst % 2], 1)
    PD = G.proj(Dm)
    Pi = np.eye(d) - PD
    hs, ns, zL, yN = G.forward_N(net, norm)

    # A. Fact F14
    A = rng.normal(size=(d, d)) + 2 * np.eye(d)
    _, yA, _ = fwd_general(net, norm, A)
    if not norm:
        check("A-F14-nonorm", np.abs(yA - yN).max() < 1e-9, np.abs(yA - yN).max())
    else:
        check("A-F14-norm-changes", np.abs(yA - yN).max() > 1e-6)
        Ai = np.linalg.inv(A)
        z1 = G.nrm(net["e"] @ A.T, net["eps"][0], True) @ (net["R"][0] @ Ai).T
        s1 = np.sqrt(((net["e"] ** 2).sum(-1) + net["eps"][0]) / (((net["e"] @ A.T) ** 2).sum(-1) + net["eps"][0]))
        check("A-F14-scale", np.abs(z1 - s1[:, None] * (ns[0] @ net["R"][0].T)).max() < 1e-9)
        Qo, _ = np.linalg.qr(rng.normal(size=(d, d)))
        _, yO, _ = fwd_general(net, norm, Qo)
        check("A-F14-orth", np.abs(yO - yN).max() < 1e-9)

    cvals = [float((p * ((hh @ PD) ** 2).sum(-1)).sum()) for hh in hs]
    epsc = [net["eps"][l] + cvals[l] for l in range(L + 1)] if norm else list(net["eps"])
    Rp, bp, Up, cyp = [r @ Pi for r in net["R"]], list(net["b"]), net["U"] @ Pi, net["cy"]
    Rs, bs, Us, cys, best = tv9_cut(net, Pi, p, norm, hs, ns, zL, epsc)

    # B. Lemma TV10 (no norm): oblique cut along a random complement K
    if not norm:
        Kb = Pi @ rng.normal(size=(d, d - Dm.shape[1])) + Dm @ rng.normal(size=(Dm.shape[1], d - Dm.shape[1]))
        M = np.concatenate([Kb, Dm], 1)
        PiK = M @ np.diag([1.0] * Kb.shape[1] + [0.0] * Dm.shape[1]) @ np.linalg.inv(M)
        J = PiK  # J on D perp; J^(-1) on K is Pi restricted to K (Pi PiK = Pi on D perp)
        Jinv = Pi  # Pi J z = Pi PiK z = z for z in D perp
        check("B-TV10-J", np.abs(Jinv @ PiK @ Pi - Pi).max() < 1e-9)
        Ro = [r @ Jinv for r in Rs]    # r'' . (J z) = r' . Pi J z = r' . z
        _, yS = cut_run(net, Rs, bs, Us, cys, epsc, Pi, False)
        _, yO = cut_run(net, Ro, bs, Us @ Jinv, cys, epsc, PiK, False)
        check("B-TV10", np.abs(yS - yO).max() < 1e-8, np.abs(yS - yO).max())

    # C. Lemma TV11
    E = areas[0] + 0.7 * areas[3]          # an area at an angle to D (meets D only in 0)
    r = rng.normal(size=d)
    Mde = np.concatenate([Dm, E], 1)
    target = np.concatenate([np.zeros(Dm.shape[1]), E.T @ r])
    rp = np.linalg.lstsq(Mde.T, target, rcond=None)[0]
    check("C-TV11-1", np.abs(Dm.T @ rp).max() < 1e-9 and np.abs(E.T @ rp - E.T @ r).max() < 1e-9)
    Eb = np.concatenate([areas[0], Dm[:, :1]], 1)   # shares the vector Dm[:, 0] with D
    Mde = np.concatenate([Dm, Eb], 1)
    target = np.concatenate([np.zeros(Dm.shape[1]), Eb.T @ r])
    rp = np.linalg.lstsq(Mde.T, target, rcond=None)[0]
    check("C-TV11-2", np.abs(Mde.T @ rp - target).max() > 1e-6 or abs(r @ Dm[:, 0]) < 1e-9)

    # D. Lemma TV12 (norm form): constants eps + c against N's constants and one more direction
    if norm:
        _, y1 = cut_run(net, Rp, bp, Up, cyp, epsc, Pi, True)
        _, y2 = cut_run(net, Rp, bp, Up, cyp, list(net["eps"]), Pi, True, battery=cvals)
        check("D-TV12", np.abs(y1 - y2).max() < 1e-9, np.abs(y1 - y2).max())

    # E. Theorem TV9: least leak
    lk_s = leaks(net, Rs, bs, Pi, norm, hs, ns, epsc)
    lk_p = leaks(net, Rp, bp, Pi, norm, hs, ns, epsc)
    for l in range(L):
        vs, vp = p @ lk_s[l] ** 2, p @ lk_p[l] ** 2
        check("E-TV9-vs-proj", (vs <= vp * (1 + 1e-9) + 1e-12).all())
        check("E-TV9-value", np.allclose(vs, best[l], rtol=1e-6, atol=1e-10), (vs, best[l]))
        check("E-TV9-perp", np.abs(Rs[l] @ PD).max() < 1e-8)
        for _ in range(5):
            Rr = Rs[l] + 0.1 * rng.normal(size=Rs[l].shape) @ Pi
            br = bs[l] + 0.1 * rng.normal(size=q)
            z = G.nrm(hs[l] @ Pi.T, epsc[l], norm)
            lr = z @ Rr.T + br - (ns[l] @ net["R"][l].T + net["b"][l])
            check("E-TV9-random", (vs <= p @ lr ** 2 * (1 + 1e-9) + 1e-12).all())

    # TV6 bounds and losses for both cuts; TV9.2 lower bound
    out = {}
    for name, (Rc, bc, Uc, cyc) in (("proj", (Rp, bp, Up, cyp)), ("B*", (Rs, bs, Us, cys))):
        _, yB = cut_run(net, Rc, bc, Uc, cyc, epsc, Pi, norm)
        dp = G.kl(p, yN, yB)
        bd = tv6_bound(net, Rc, bc, Uc, cyc, epsc, Pi, areas, p, norm)
        check(f"F-TV6-holds-{name}", dp <= bd * (1 + 1e-9) + 1e-14, (dp, bd))
        ratios[name].append(bd / max(dp, 1e-300))
        out[name] = dp
    lb = tv92_bound(p, yN, net["e"] @ Pi.T)
    check("F-TV9.2-dense", lb <= min(out.values()) + 1e-12)
    loss_split.append((out["proj"], out["B*"]))

    # G. Lemma TV13 (no norm): the signed identity, and where TV6 loses against it
    if not norm:
        for name, (Rc, bc, Uc, cyc) in (("proj", (Rp, bp, Up, cyp)), ("B*", (Rs, bs, Us, cys))):
            hc, yB = cut_run(net, Rc, bc, Uc, cyc, epsc, Pi, False)
            lk = leaks(net, Rc, bc, Pi, False, hs, ns, epsc)
            Js, terms = [], []
            for l in range(L):
                a = hs[l] @ net["R"][l].T + net["b"][l]
                ap = hc[l] @ Rc[l].T + bc[l]
                da = ap - a
                sl = np.where(np.abs(da) > 1e-14, (G.act(ap, kind) - G.act(a, kind)) / np.where(np.abs(da) > 1e-14, da, 1), 0)
                Js.append((sl, Rc[l]))
            total = ((hs[L] @ Pi.T) @ Uc.T + cyc - hs[L] @ net["U"].T - net["cy"]) @ Pperp.T   # omega
            per_l = []
            for l in range(L):
                v = (lk[l] * Js[l][0]) @ (Pi @ net["W"][l]).T            # Pi W_l S_l leak_l, NX x d
                for m in range(l + 1, L):
                    sl, Rm = Js[m]
                    v = v + ((v @ Rm.T) * sl) @ (Pi @ net["W"][m]).T     # (I + J_m) v
                per_l.append(v @ Uc.T @ Pperp.T)
            total = total + sum(per_l)
            left = (yB - yN) @ Pperp.T
            check(f"G-TV13-{name}", np.abs(left - total).max() < 1e-8, np.abs(left - total).max())
            if name == "proj":
                om = ((hs[L] @ Pi.T) @ Uc.T + cyc - hs[L] @ net["U"].T - net["cy"]) @ Pperp.T
                w = lambda f: np.sqrt(float((p * (f ** 2).sum(-1)).sum()))
                exact = w(left)
                mink_l = w(om) + sum(w(t_) for t_ in per_l)          # signs kept inside each block's term
                bd = tv6_bound(net, Rc, bc, Uc, cyc, epsc, Pi, areas, p, False)
                absl = 0.0   # every entry by its size, true slopes: only the signs are dropped
                PUa = np.abs(Pperp @ Uc)
                for l in range(L):
                    v = (np.abs(lk[l]) * np.abs(Js[l][0])) @ np.abs(Pi @ net["W"][l]).T
                    for m in range(l + 1, L):
                        sl, Rm = Js[m]
                        v = v + ((v @ np.abs(Rm).T) * np.abs(sl)) @ np.abs(Pi @ net["W"][m]).T
                    absl += w(v @ PUa.T)
                absl += w(om)
                gap_parts.append((exact, mink_l, 2 * np.sqrt(bd), 0.25 * exact ** 2 / max(out["proj"], 1e-300), absl))

    # H. Proposition TV14 (no norm, tanh): signed carrying with N's own slopes, pointwise remainder
    if not norm and kind == "tanh":
        for name, (Rc, bc, Uc, cyc) in (("proj", (Rp, bp, Up, cyp)), ("B*", (Rs, bs, Us, cys))):
            _, yB = cut_run(net, Rc, bc, Uc, cyc, epsc, Pi, False)
            Lam, rem, dp = tv14(net, Rc, bc, Uc, cyc, Pi, p, hs, yN, yB)
            left = (yB - yN) @ Pperp.T
            check(f"H-TV14-pointwise-{name}", (np.linalg.norm(left - Lam, axis=1) <= rem * (1 + 1e-9) + 1e-12).all())
            w = lambda f: np.sqrt(float((p * (f ** 2 if f.ndim == 1 else (f ** 2).sum(-1))).sum()))
            b14 = 0.25 * (w(Lam) + w(rem)) ** 2
            check(f"H-TV14-bound-{name}", dp <= b14 * (1 + 1e-9) + 1e-14, (dp, b14))
            tv14_ratios.append(b14 / max(dp, 1e-300))
        if len(scale_rows) < 3:   # the family N_t: N's reads of D (blocks and logits) times t; projection cut
            row = []
            for t_ in (1.0, 0.3, 0.1, 0.03, 0.01):
                nt = dict(net, R=[r @ Pi + t_ * r @ PD for r in net["R"]], U=net["U"] @ Pi + t_ * net["U"] @ PD)
                hst, nst, zLt, yNt = G.forward_N(nt, False)
                Rpt, Upt = [r @ Pi for r in nt["R"]], nt["U"] @ Pi
                _, yBt = cut_run(nt, Rpt, bp, Upt, cyp, epsc, Pi, False)
                Lam, rem, dp = tv14(nt, Rpt, bp, Upt, cyp, Pi, p, hst, yNt, yBt)
                left = (yBt - yNt) @ Pperp.T
                check("H-TV14-family", (np.linalg.norm(left - Lam, axis=1) <= rem * (1 + 1e-9) + 1e-12).all())
                b14 = 0.25 * (np.sqrt(p @ (Lam ** 2).sum(-1)) + np.sqrt(p @ rem ** 2)) ** 2
                b6 = tv6_bound(nt, Rpt, bp, Upt, cyp, epsc, Pi, areas, p, False)
                f11 = 0.25 * float(p @ (left ** 2).sum(-1))
                row.append((t_, b6 / dp, b14 / dp, f11 / dp, float(np.sqrt(p @ rem ** 2) / np.sqrt(p @ (Lam ** 2).sum(-1)))))
            scale_rows.append(row)

# F. Proposition TV9.1 (the copy network) and TV9.2 on TV7's network
xs = np.array([-1.0, 1.0]); w2 = np.array([.5, .5]); E3 = np.eye(3)
e = xs[:, None] * (E3[0] + E3[1])
r1, w1 = E3[1], E3[2]
h1 = e + (e @ r1)[:, None] * w1
U = np.stack([s_ * E3[2], np.zeros(3)])
yN = h1 @ U.T
Pi = np.eye(3) - np.outer(E3[1], E3[1])
h1p = (e @ Pi.T) + ((e @ Pi.T) @ (Pi @ r1))[:, None] * w1        # projection cut, reads Pi r
yP = (h1p @ Pi.T) @ (U @ Pi).T
H = -float((qp * np.log(qp)).sum())
check("F-TV9.1-proj", abs(G.kl(w2, yN, yP) - (np.log(2) - H)) < 1e-12, (G.kl(w2, yN, yP), np.log(2) - H))
rstar = E3[0]                                                      # t = x = e1 . Pi h0
check("F-TV9.1-leak0", np.abs((e @ Pi.T) @ rstar - e @ r1).max() < 1e-15)
h1s = (e @ Pi.T) + ((e @ Pi.T) @ rstar)[:, None] * w1
Us = U @ Pi                                                        # reads e3 = what N reads
check("F-TV9.1-exact", G.kl(w2, yN, h1s @ Us.T) < 1e-15)
P4 = np.array([[a, b] for a in (-1, 1) for b in (-1, 1)], float)
y7 = np.stack([s_ * P4[:, 1], np.zeros(4)], 1)
check("F-TV9.2-TV7", abs(tv92_bound(np.ones(4) / 4, y7, P4[:, :1]) - c_s) < 1e-12)

print(f"instances {N_INST}; failures {len(fails)}")
for f in fails[:20]:
    print("  FAIL", f)
for k, v in ratios.items():
    v = np.array(v)
    print(f"TV6 bound / D_P, {k:5s}: median {np.median(v):.3g}, range {v.min():.3g} to {v.max():.3g}")
ls = np.array(loss_split)
print(f"D_P projection cut / D_P of B*: median {np.median(ls[:, 0] / ls[:, 1]):.3g}, "
      f"B* lower in {int((ls[:, 1] < ls[:, 0]).sum())} of {len(ls)}")
gp = np.array(gap_parts)
print(f"no norm, projection cut, norms of the logit change: exact {np.median(gp[:, 0]):.3g}; signs kept per block "
      f"(Minkowski over blocks) {np.median(gp[:, 1]):.3g}; TV6 {np.median(gp[:, 2]):.3g} (medians); "
      f"median ratios Minkowski/exact {np.median(gp[:, 1] / gp[:, 0]):.3g}, TV6/Minkowski {np.median(gp[:, 2] / gp[:, 1]):.3g}; "
      f"softmax step (F11: exact^2 / 4 over D_P) {np.median(gp[:, 3]):.3g}; TV6 bound / D_P here {np.median((gp[:, 2] / 2) ** 2 / (gp[:, 0] ** 2 / 4 / gp[:, 3])):.3g}")
print(f"  inside TV6/Minkowski: signs dropped (entrywise sizes, true slopes) / Minkowski {np.median(gp[:, 4] / gp[:, 1]):.3g}; "
      f"TV6 / signs dropped {np.median(gp[:, 2] / gp[:, 4]):.3g} (slopes set to Lip, sizes per area)")
t14 = np.array(tv14_ratios)
print(f"TV14 bound / D_P (no norm, tanh, projection cut and B*): median {np.median(t14):.3g}, range {t14.min():.3g} to {t14.max():.3g}")
for row in scale_rows:
    print("  family N_t (t, TV6/D_P, TV14/D_P, F11 step, rem/Lambda): " +
          "; ".join(f"{t_:g}: {a:.3g}, {b:.3g}, {c:.3g}, {r:.2g}" for t_, a, b, c, r in row))
