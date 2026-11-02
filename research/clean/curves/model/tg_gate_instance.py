# serves: D34
"""Exact test of Conjecture D34 (COARSE-THEORY.md D.11) and a check of Lemma T39 (B.G), on toy networks only.

N: layer 1 h = (relu(x1), relu(x2), relu(-x1), relu(-x2), relu(x3)); unit 5 is the gate g. Layer 2: members
relu(h_i + 5 g - 6), i = 1..4; N's gap = their sum. Points with x3 = 1 are in the gate's region G, x3 = 0 off it.
Builds keep layer 1 (Lemma T39 part 0: the problem is the one-hidden-layer problem on the points z = h(x)).
Test: global least width-1 value (exact rule of T21, trunk_exact.exact_min) against the least gate fold (the same
piece solver with the unit's region fixed to G, both signs). Also Lemma T39: L_G from one weighted fit on G, and the
part-3 sign condition.
Predictions (written before the run, in the paper): D34 predicts equality on I1 and I2, both weights, D2 and KL.
My expectation: I2 may fail (detector-like inside the gate).
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
from trunk_exact import exact_min, per_point_loss, solve_piece  # noqa: E402


def orbit(p):
    out, q = [], tuple(p)
    for _ in range(4):
        out.append(q)
        q = (-q[1], q[0])
    return out


def h(x1, x2, x3):
    r = lambda v: max(v, 0.0)
    return np.array([r(x1), r(x2), r(-x1), r(-x2), r(x3)])


def gap_N(z):
    return sum(max(z[i] + 5 * z[4] - 6, 0.0) for i in range(4))


def instance(region_orbits, extra_region, w0):
    pts = [(a, b, 1) for o in region_orbits for (a, b) in orbit(o)] + [(a, b, 1) for (a, b) in extra_region]
    pts = pts + [(0, 0, 0)]
    Z = np.array([h(*p) for p in pts])
    nG = len(pts) - 1
    pi = np.array([(1 - w0) / nG] * nG + [w0])
    t = np.array([gap_N(z) for z in Z])
    G = frozenset(range(nG))
    return pts, Z, pi, t, G


def t39_fit(Z, pi, t, G, kind):
    """Lemma T39: the least affine fit of t on G (values F), and L_G."""
    idx = sorted(G)
    A = np.hstack([Z[idx], np.ones((len(idx), 1))])
    w = pi[idx]
    lossf, gradf = per_point_loss(kind, t[idx])
    f = lambda c: float(np.sum(w * lossf(A @ c)))
    df = lambda c: A.T @ (w * gradf(A @ c))
    c0 = np.linalg.lstsq(A * np.sqrt(w)[:, None], t[idx] * np.sqrt(w), rcond=None)[0]
    r = minimize(f, c0, jac=df, method="BFGS", options={"gtol": 1e-12})
    return A @ r.x, r.fun


def run(name, region_orbits, extra_region, w0, kind):
    pts, Z, pi, t, G = instance(region_orbits, extra_region, w0)
    (v_all, z_all, ch_all), Xt = exact_min(Z, pi, t, 1, kind=kind, starts=4)
    lossf, gradf = per_point_loss(kind, t)
    gate = [solve_piece(Xt, pi, lossf, gradf, [(s, G)], starts=6) for s in (1, -1)]
    v_gate = min(g[0] for g in gate)
    F, L_G = t39_fit(Z, pi, t, G, kind)
    delta0 = 0.0
    sign_ok = bool(np.all(F - delta0 > 1e-9) or np.all(F - delta0 < -1e-9))
    s, A = ch_all[0]
    D = Xt.shape[1]
    th = z_all[:D]
    pre = Xt @ th
    return {
        "instance": name, "w0": w0, "kind": kind, "t": t.tolist(),
        "least_all": v_all, "least_gate_fold": v_gate, "L_G_formula": L_G,
        "T39_part3_sign_condition": sign_ok, "fitted_on_G": F.round(6).tolist(),
        "least_region": [int(i) for i in sorted(A)], "least_sign": int(s), "least_unit": th.round(6).tolist(), "least_gamma": float(z_all[-1]),
        "least_region_is_G": frozenset(A) == G,
        "least_preact": pre.round(6).tolist(),
        "rel_gap": (v_gate - v_all) / max(v_all, 1e-15),
        "points": [list(p) for p in pts],
    }


def main():
    I1 = ([(2, 0), (2, 2), (1, 1)], [])
    I2 = ([(2, 0), (1, 1)], [(0, 0)])
    out = []
    for name, (orbs, extra) in (("I1", I1), ("I2", I2)):
        nG = 4 * len(orbs) + len(extra)
        for w0 in (1 / (nG + 1), 0.5):
            for kind in ("d2", "kl"):
                r = run(name, orbs, extra, w0, kind)
                out.append(r)
                print(f"{name} w0={w0:.3f} {kind}: all {r['least_all']:.8f}  gate-fold {r['least_gate_fold']:.8f}  "
                      f"L_G {r['L_G_formula']:.8f}  sign-ok {r['T39_part3_sign_condition']}  "
                      f"region=G {r['least_region_is_G']}  region {r['least_region']} sign {r['least_sign']}")
    with open(os.path.join(os.path.dirname(__file__), "tg_gate_instance.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
