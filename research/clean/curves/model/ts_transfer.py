# serves: T42
"""Check of Theorem T42 (COARSE-THEORY.md B.S) and Remark B.S.1 on toy networks, objective D2, width 1.

Teachers: T20's bathtub (points -3..3, gap relu(x - 1) + relu(-x - 1), uniform) and instance I1 of B.G (on the points
z = h(x), Lemma T39 part 0). For each teacher t and 40 random perturbations t' = t + noise (scale 1/10 to 1; every
vector on a finite mix is the gap vector of some network, so t' is a network N'), with dist = ||t - t'||_P:
  part 1: |delta(t) - delta(t')| <= dist;
  part 2: the least build B' for t' satisfies ||gap_B' - t||_P <= delta(t) + 2 dist.
Here ||v||_P^2 = sum_i pi_i v_i^2 / 2 (the centred norm for two logits) and delta = sqrt(2 D2*).
Least builds come from the exact rule of T21 (trunk_exact.exact_min), accurate to the solver tolerance, so each check
allows a slack of 1/10^5.
Remark B.S.1 on I1: count the optimal (sign, region) pieces. Prediction: at least two, an orbit of the rotation
group (B* and its turn by 90 degrees).
Predictions (written before the run): parts 1 and 2 hold in every trial; the largest ratio of the part-2 excess
(||gap_B' - t|| - delta(t)) to dist is at most 2.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from trunk_exact import exact_min, per_point_loss, regions, solve_piece  # noqa: E402


def norm(v, pi):
    return float(np.sqrt(np.sum(pi * v ** 2) / 2))


def least(X, pi, t):
    (v, z, ch), Xt = exact_min(X, pi, t, 1, kind="d2", starts=3)
    D = Xt.shape[1]
    s, _ = ch[0]
    gap = s * np.maximum(Xt @ z[:D], 0) + z[-1]
    return v, gap


def bathtub():
    x = np.arange(-3, 4, dtype=float)[:, None]
    t = np.maximum(x[:, 0] - 1, 0) + np.maximum(-x[:, 0] - 1, 0)
    return x, np.full(7, 1 / 7), t


def i1():
    def orbit(p):
        out, q = [], tuple(p)
        for _ in range(4):
            out.append(q)
            q = (-q[1], q[0])
        return out
    pts = [(a, b, 1) for o in [(2, 0), (2, 2), (1, 1)] for (a, b) in orbit(o)] + [(0, 0, 0)]
    r = lambda v: max(v, 0.0)
    Z = np.array([[r(a), r(b), r(-a), r(-b), r(c)] for a, b, c in pts])
    t = np.array([sum(r(z[i] + 5 * z[4] - 6) for i in range(4)) for z in Z])
    return Z, np.full(13, 1 / 13), t


def main():
    rng = np.random.default_rng(0)
    out = {"trials": []}
    worst1, worst2 = -np.inf, 0.0
    ok = True
    for name, (X, pi, t) in (("bathtub", bathtub()), ("I1", i1())):
        v0, g0 = least(X, pi, t)
        d0 = np.sqrt(2 * v0)
        assert abs(d0 - norm(g0 - t, pi)) < 1e-5
        for trial in range(40):
            scale = rng.uniform(0.1, 1.0)
            tp = t + scale * rng.normal(size=t.shape)
            dist = norm(t - tp, pi)
            vp, gp = least(X, pi, tp)
            dp = np.sqrt(2 * vp)
            p1 = abs(d0 - dp) - dist
            excess = norm(gp - t, pi) - d0
            p2 = excess - 2 * dist
            ok &= bool(p1 <= 1e-5 and p2 <= 1e-5)
            worst1 = max(worst1, p1)
            worst2 = max(worst2, excess / dist)
            out["trials"].append({"teacher": name, "dist": dist, "part1_slack": p1, "part2_slack": p2,
                                  "excess_over_dist": excess / dist})
        print(f"{name}: delta = {d0:.6f}")
    out["all_hold"] = ok
    out["max_part1_slack"] = worst1
    out["max_excess_over_dist"] = worst2
    print("parts 1 and 2 hold in every trial:", ok)
    print("largest |delta - delta'| - dist:", worst1, " largest excess / dist:", worst2)

    # Remark B.S.1 on I1: optimal pieces
    X, pi, t = i1()
    regs, Xt = regions(X)
    lossf, gradf = per_point_loss("d2", t)
    vals = []
    for A in regs:
        for s in (1, -1):
            v, _ = solve_piece(Xt, pi, lossf, gradf, [(s, A)], starts=3)
            vals.append((v, s, sorted(int(i) for i in A)))
    vmin = min(v for v, _, _ in vals)
    opt = [(s, A) for v, s, A in vals if v <= vmin + 1e-7]
    out["I1_least"] = vmin
    out["I1_optimal_pieces"] = opt
    print(f"I1: least {vmin:.8f} (5/182 = {5/182:.8f}); optimal pieces: {len(opt)}")
    for s, A in opt:
        print("  sign", s, "region", A)
    with open(os.path.join(os.path.dirname(__file__), "ts_transfer.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
