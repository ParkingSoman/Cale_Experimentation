# serves: TQ31
"""Step 2. Rebuild hard tables with other units: same logits on P, then the principle errors.

Builds, fixed before the run:
 A. TQ31's table. One hidden layer. Area unit h_w (votes only that live classes beat dead ones) and two specialists u+_w, u-_w, never on together.
 B. TQ31's table. Two hidden layers: a gate and two point units in front of A's units, so G and B are not empty.
 E. Square family table. Layer 1: v_a, family gate Phi_v, poles P+_v, P-_v. Layer 2: J_a, coarse reader R_v, Q+_v, Q-_v. Units Q, R are shared by the t rows of a family.
 F. Square table with holes M = sig sig^T (sig a two-siding): poles shared by ALL areas of a camp, levels in per-family units.
PREDICTION: A, B, E: eps_V = 0, G = B = 0, pack = 0, unread = 0. Agreement: unknown. F: literal eps_V > 0, since a shared unit must vote against some row's levels. It reuses most and scores worst.
"""
import sys

import numpy as np

sys.path.insert(0, "/Users/shonusengupta/Cale_Experimentation/research/clean/curves/rounds/round4/i5")
from netlib import *
from sqlib import *


def cols_live(yp, ym, kappa_=.25):
    """col_h = kappa on classes above the dead level in both tables."""
    live = (np.minimum(yp, ym) > 1e-9)
    return kappa_ * live


def build_A(y0, mv):
    m, C = y0.shape
    W1 = np.zeros((3 * m, 2 * m))
    b1 = np.zeros(3 * m)
    Wout = np.zeros((C, 3 * m))
    for w in range(m):
        W1[w, w] = 1
        W1[m + w, [w, m + w]] = -2, 1
        W1[2 * m + w, [w, m + w]] = 2, -1
        ch = cols_live(y0[w] + mv[w], y0[w] - mv[w])
        Wout[:, w] = ch
        Wout[:, m + w] = y0[w] + mv[w] - ch
        Wout[:, 2 * m + w] = y0[w] - mv[w] - ch
    return [(W1, b1)], Wout, np.zeros(C)


def build_B(y0, mv):
    [(W1, b1)], Wout, c = build_A(y0, mv)
    m = y0.shape[0]
    W2 = np.zeros((3 * m, 3 * m))
    b2 = np.zeros(3 * m)
    for w in range(m):
        W2[w, w] = 1
        W2[m + w, [m + w, w]] = 1, 1
        b2[m + w] = -1
        W2[2 * m + w, [2 * m + w, w]] = 1, 1
        b2[2 * m + w] = -1
    return [(W1, b1), (W2, b2)], Wout, c


def build_E(M, W, fam, a, kappa_=.5):
    n = len(W)
    q = fam.max() + 1
    y0 = np.log(W)
    # layer 1: v_a (n), Phi_v (q), P+_v (q), P-_v (q). inputs: h coords 0..n-1, s coords n..2n-1
    W1 = np.zeros((n + 3 * q, 2 * n))
    for i in range(n):
        W1[i, i] = 1
        v = fam[i]
        W1[n + v, i] = 1
        W1[n + q + v, [i, n + i]] = -2, 1
        W1[n + 2 * q + v, [i, n + i]] = 2, -1
    b1 = np.zeros(n + 3 * q)
    # layer 2: J_a (n), R_v (q), Q+_v (q), Q-_v (q)
    W2 = np.zeros((n + 3 * q, n + 3 * q))
    b2 = np.zeros(n + 3 * q)
    for i in range(n):
        W2[i, i] = 1
    for v in range(q):
        mem_ = np.where(fam == v)[0]
        # v_a is also read by the family's shared units, so that no unit is read by a reader that reads it alone (Lemma T46c)
        # Phi_v has the largest vote in every shared unit, so each shared unit has one route on its whole target
        W2[n + v, n + v] = 2
        W2[n + v, mem_] = 1
        b2[n + v] = -2
        W2[n + q + v, [n + q + v, n + v]] = 1, 2
        W2[n + q + v, mem_] = 1
        W2[n + 2 * q + v, [n + 2 * q + v, n + v]] = 1, 2
        W2[n + 2 * q + v, mem_] = 1
        b2[n + q + v] = b2[n + 2 * q + v] = -3
    Wout = np.zeros((n, n + 3 * q))
    for v in range(q):
        mem = np.where(fam == v)[0]
        rep = mem[0]
        l = np.log(W[rep, mem[1]]) if len(mem) > 1 else np.log(W[rep].copy()[np.argsort(W[rep])[-2]]) + a + 1
        yp = y0[rep] + a / 2 * M[rep]
        ym = y0[rep] - a / 2 * M[rep]
        cp, cm = yp.copy(), ym.copy()
        cp[mem] = l + a / 2 - kappa_
        cm[mem] = l - a / 2 - kappa_
        Wout[:, n + q + v] = cp
        Wout[:, n + 2 * q + v] = cm
        Wout[mem, n + v] = kappa_
        for i in mem:
            Wout[i, i] = y0[i, i] - l
    return [(W1, b1), (W2, b2)], Wout, np.zeros(n)


def build_F(M, W, fam, sig, a):
    """Two poles shared by all areas of the network: C+ on at x^+ of camp + and x^- of camp -, C- the reverse. Exceptions on rows with an uncut side link. Levels in per-area units."""
    n = len(W)
    q = fam.max() + 1
    s = sig[fam]
    y0 = np.log(W)
    X = np.outer(s, s)
    E = M - X  # exceptions
    rows = np.where(np.abs(E).sum(1) > 0)[0]
    nu = 2 * n + 2 + 2 * len(rows)  # h_a, spare, C+, C-, exception poles
    W1 = np.zeros((n + 2 + 2 * len(rows), 2 * n))
    for i in range(n):
        W1[i, i] = 1
        W1[n, [i, n + i]] = -2 * s[i], s[i]
        W1[n + 1, [i, n + i]] = 2 * s[i], -s[i]
    Wout = np.zeros((n, W1.shape[0]))
    Wout[:, :n] = y0.T
    Wout[:, n] = a / 2 * s
    Wout[:, n + 1] = -a / 2 * s
    for k, i in enumerate(rows):
        W1[n + 2 + 2 * k, [i, n + i]] = -2, 1
        W1[n + 3 + 2 * k, [i, n + i]] = 2, -1
        Wout[:, n + 2 + 2 * k] = a / 2 * E[i]
        Wout[:, n + 3 + 2 * k] = -a / 2 * E[i]
    return [(W1, np.zeros(W1.shape[0]))], Wout, np.zeros(n)


def report(tag, layers, Wout, c, Ytrue, m):
    X = points(m)
    p = np.ones(2 * m) / (2 * m)
    err = np.abs(forward(layers, Wout, c, X)[2] - Ytrue).max()
    r = principles(layers, Wout, c, X, p)
    per = r["per"]
    badxi = [(q["layer"], q["unit"], round(q["xi"], 5)) for q in per if q["xi"] is not None and q["xi"] <= 1e-13]
    print(f"{tag}: units {r['units']} table_err={err:.1e} eps_V={r['eps_V']:.4f} V_rel={r['eps_V_rel']:.4f} G={r['eps_G']} B={r['eps_B']} pack={r['eps_pack']:.4f} unread={r['eps_unread']:.3f} agree={r['eps_agree']:.3f} (units with xi<=0: {len(badxi)} of {len(per)}) min_lam={min(q['lam'] for q in per):.4f} median|xi|={np.median([abs(q['xi']) for q in per]):.2e}", flush=True)
    return r


for a, rho in ((1.0, .31), (4.0, .31)):
    print(f"--- a={a} rho={rho}")
    for name, (n, e) in {"C5": cycle(5), "sub(K3)": subdivide(*complete(3))}.items():
        U, sig = least_uncut(n, e)
        D = float(np.exp(np.ceil(np.log(n ** 2 * 50))))
        y0, mv = tq31(n, e, a, rho, D)
        Y = table(y0, mv)
        lay, Wo, c = net_dq18(y0, mv)
        report(f"N  {name} U={U}", lay, Wo, c, Y, n)
        report(f"A  {name}", *build_A(y0, mv), Y, n)
        report(f"B  {name}", *build_B(y0, mv), Y, n)
    for name, (qq, e), t in (("C5 t=2", cycle(5), 2), ("C5 t=3", cycle(5), 3), ("C4 t=2", cycle(4), 2)):
        U, sig = least_uncut(qq, e)
        n = qq * t
        D = float(np.exp(np.ceil(np.log(n ** 2 * 50))))
        M, W, fam, adj, Df = sqfam(qq, e, t, a, rho, D)
        y0 = np.log(W)
        Y = table(y0, a / 2 * M)
        s = softmax(Y)
        print(f"sq {name}: n={n} U={U} own mass {s[np.arange(0, 2 * n, 2), np.repeat(np.arange(n), 1)].mean():.3f}/{s[np.arange(1, 2 * n, 2), np.arange(n)].mean():.3f} sib mass {(s[1] * (fam == fam[0]) ).sum() - s[1, 0]:.3f} side mass {(s[1] * adj[fam[0]][fam]).sum():.3f}")
        report(f"N-form sq {name}", *net_dq18(y0, a / 2 * M), Y, n)
        report(f"E {name}", *build_E(M, W, fam, a), Y, n)
    qq, e = cycle(5)
    t = 2
    n = qq * t
    U, sig = least_uncut(qq, e)
    D = float(np.exp(np.ceil(np.log(n ** 2 * 50))))
    M, W, fam, adj, Df = sqfam(qq, e, t, a, rho, D, hole_sign=sig)
    Y = table(np.log(W), a / 2 * M)
    report(f"F camp poles C5 t=2 U={U}", *build_F(M, W, fam, sig, a), Y, n)
