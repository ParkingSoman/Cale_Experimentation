# serves: D59
"""Part B.X (COARSE-THEORY.md): kink profiles, in exact fractions.

1. Theorem T63: on random rational ReLU networks with two and three hidden layers, along random rational lines, the
   lambda-dip (lambda = 1/2 and 1/3) equals -sum jump_j K_lambda(t_j), and every kink is a zero of some unit.
2. Proposition T66: the tent-map network with k layers of two units is exact on the grid j / 2^k (k = 1..6), and the
   grid has 2^(k-1) - 1 strict dip lines with disjoint open segments.
3. Kink profiles: T20 (all lines with midpoint in P) and T61's detector instance (its strict dip lines, and which units
   of the clique build and of T61's build switch across each, with their jumps).
"""
import itertools
import json
import os
import random
from fractions import Fraction as Fr

R = random.Random(63)


# ---------- piecewise-linear functions of t on [0, 1], as sorted knots (t, value) ----------
def pl_const_line(a, b):
    return [(Fr(0), a), (Fr(1), b)]


def pl_eval(f, t):
    for (t0, v0), (t1, v1) in zip(f, f[1:]):
        if t0 <= t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    raise ValueError


def pl_affine(fs, coefs, const):
    ts = sorted({t for f in fs for t, _ in f})
    return [(t, sum(c * pl_eval(f, t) for f, c in zip(fs, coefs)) + const) for t in ts]


def pl_relu(f):
    out = []
    for (t0, v0), (t1, v1) in zip(f, f[1:]):
        out.append((t0, max(v0, Fr(0))))
        if (v0 > 0 > v1) or (v0 < 0 < v1):
            tz = t0 + (t1 - t0) * v0 / (v0 - v1)
            out.append((tz, Fr(0)))
    out.append((f[-1][0], max(f[-1][1], Fr(0))))
    return out


def kinks(f):
    out = []
    for (ta, va), (tb, vb), (tc, vc) in zip(f, f[1:], f[2:]):
        s1 = (vb - va) / (tb - ta)
        s2 = (vc - vb) / (tc - tb)
        if s1 != s2:
            out.append((tb, s2 - s1))
    return out


def K(lam, t):
    return min(t * (1 - lam), lam * (1 - t))


def zeros(f):
    zs = set()
    for (t0, v0), (t1, v1) in zip(f, f[1:]):
        if v0 == 0:
            zs.add(t0)
        if (v0 > 0 > v1) or (v0 < 0 < v1):
            zs.add(t0 + (t1 - t0) * v0 / (v0 - v1))
    if f[-1][1] == 0:
        zs.add(f[-1][0])
    return zs


def rq(n=3, den=2):
    return Fr(R.randint(-n * den, n * den), den)


def random_net(d, widths):
    layers, prev = [], d
    for w in widths:
        layers.append(([[rq() for _ in range(prev)] for _ in range(w)], [rq() for _ in range(w)]))
        prev = w
    return layers, [rq() for _ in range(prev)], rq()


def gap_along(net, x, xp):
    layers, a, c = net
    h = [pl_const_line(xi, xpi) for xi, xpi in zip(x, xp)]
    unit_zero_sets = set()
    for W, b in layers:
        pres = [pl_affine(h, row, bi) for row, bi in zip(W, b)]
        for p in pres:
            unit_zero_sets |= zeros(p)
        h = [pl_relu(p) for p in pres]
    return pl_affine(h, a, c), unit_zero_sets


def check_t63(trials=300):
    ok, n_kinks = True, 0
    for _ in range(trials):
        d = R.choice([2, 3])
        net = random_net(d, R.choice([[3, 3], [4, 3], [3, 3, 2], [4, 4, 3]]))
        x = [rq() for _ in range(d)]
        xp = [rq() for _ in range(d)]
        g, zs = gap_along(net, x, xp)
        ks = [(t, j) for t, j in kinks(g) if 0 < t < 1]
        n_kinks += len(ks)
        ok &= all(t in zs for t, _ in ks)
        for lam in (Fr(1, 2), Fr(1, 3)):
            lhs = pl_eval(g, lam) - (1 - lam) * pl_eval(g, Fr(0)) - lam * pl_eval(g, Fr(1))
            rhs = -sum(j * K(lam, t) for t, j in ks)
            ok &= lhs == rhs
    return dict(trials=trials, total_kinks=n_kinks, ALL_OK=bool(ok))


def check_t66(kmax=6):
    rows, ok = [], True
    for k in range(1, kmax + 1):
        net = ([([[Fr(1)], [Fr(1)]], [Fr(0), Fr(-1, 2)])] +
               [([[Fr(2), Fr(-4)], [Fr(2), Fr(-4)]], [Fr(0), Fr(-1, 2)]) for _ in range(k - 1)], [Fr(2), Fr(-4)], Fr(0))
        layers, a, c = net
        exact = True
        for j in range(2 ** k + 1):
            h = [Fr(j, 2 ** k)]
            for W, b in layers:
                h = [max(sum(wi * hi for wi, hi in zip(row, h)) + bi, Fr(0)) for row, bi in zip(W, b)]
            gap = sum(ai * hi for ai, hi in zip(a, h)) + c
            exact &= gap == j % 2
        t = lambda j: j % 2
        dips = [j for j in range(2, 2 ** k, 2) if t(j) < Fr(t(j - 1) + t(j + 1), 2)]
        ok &= exact and len(dips) == 2 ** (k - 1) - 1
        rows.append(dict(k=k, units_deep=2 * k, exact=exact, disjoint_strict_dip_lines=len(dips)))
    return dict(rows=rows, ALL_OK=bool(ok))


def profile_t20():
    T = lambda x: max(x - 1, 0) + max(-x - 1, 0)
    P = range(-3, 4)
    lines = []
    for x, xp in itertools.combinations(P, 2):
        if (x + xp) % 2 == 0:
            mid = (x + xp) // 2
            dip = Fr(T(mid)) - Fr(T(x) + T(xp), 2)
            if dip != 0:
                lines.append(dict(x=x, xp=xp, dip=str(dip)))
    return lines


def profile_t61():
    def circle_pt(i):
        return (Fr(1 - i * i, 1 + i * i), Fr(2 * i, 1 + i * i))
    xs = [circle_pt(1), circle_pt(2)]
    r = Fr(39, 40); c = 2 / (1 - r)
    units = {"A": (0, 0), "B": (0, 1), "C": (1, 0), "D": (1, 1)}

    def vertex(u):
        j, i = units[u]; x = [Fr(0)] * 4; x[2 * j], x[2 * j + 1] = xs[i]; return x

    def z(u, x):
        j, i = units[u]; return xs[i][0] * x[2 * j] + xs[i][1] * x[2 * j + 1] - r
    bad = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D")]
    cp = Fr(4) / (1 - 1 * Fr(4, 5)) * 1  # c' = 4 / eta with eta = 1/5
    builds = {
        "clique (T59.2)": [(Fr(1), lambda x: cp * (z("A", x) + z("D", x) + 2 * r - 1) + 2),
                           (Fr(1), lambda x: cp * (z("B", x) + z("C", x) + 2 * r - 1) + 2)],
        "split dials (T61)": [(Fr(1), lambda x: -20 * z("A", x) - 40 * z("C", x) - Fr(45, 2)),
                              (Fr(1), lambda x: -40 * z("B", x) - 20 * z("D", x) - Fr(45, 2))],
    }
    out = []
    for a, b in bad:
        x, xp = vertex(a), vertex(b)
        mid = [(p + q) / 2 for p, q in zip(x, xp)]
        gN = lambda y: c * sum(max(z(u, y), Fr(0)) for u in "ABCD")
        rec = dict(line=a + "-" + b, dip=str(gN(mid) - (gN(x) + gN(xp)) / 2))
        for name, us in builds.items():
            sw = []
            for k, (ak, l) in enumerate(us):
                l0, l1 = l(x), l(xp)
                if (l0 > 0 > l1) or (l0 < 0 < l1):
                    sw.append(dict(unit=k + 1, jump=str(ak * abs(l1 - l0)), t=str(l0 / (l0 - l1))))
            rec[name] = sw
        out.append(rec)
    return out


def main():
    res = dict(t63=check_t63(), t66=check_t66(), t20_dip_lines=profile_t20(), t61_profile=profile_t61())
    print("T63:", res["t63"])
    print("T66:", [(r_["k"], r_["units_deep"], r_["exact"], r_["disjoint_strict_dip_lines"]) for r_ in res["t66"]["rows"]],
          "ALL_OK", res["t66"]["ALL_OK"])
    print("T20 strict dip lines:", [(l["x"], l["xp"], l["dip"]) for l in res["t20_dip_lines"]])
    for rec in res["t61_profile"]:
        print("T61", rec["line"], "dip", rec["dip"], "| clique:", rec["clique (T59.2)"], "| split:", rec["split dials (T61)"])
    with open(os.path.join(os.path.dirname(__file__), "tr_kinks.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
