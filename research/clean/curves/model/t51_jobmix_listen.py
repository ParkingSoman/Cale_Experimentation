# serves: T51
# serves: T51.1
# serves: T52
"""Checks of Lemma T51, Corollary T51.1 (job-weighted mixes, Definition D48) and Lemma T52 (the read tests of D49).

J1  T51: for T43 instances (kappa 2 and 3, several graphs H), the job-weighted mix P_a of layer 1 built on nu = P with
    a_(j,i) = 1/Z, a_0 = M n_bad / Z equals P exactly (Fractions), and the vector a does not depend on which pairs are bad
    beyond n_bad. Two graphs with the same n_bad give the same a and different P.
J2  T52 part 3: OR reader r = relu(h_1 + h_2), h_1 = relu(x_1), h_2 = relu(x_2), law uniform on (1,0), (0,1), (0,0):
    delta(u_1) = 1, gamma(u_1) = 0, xi(u_1) = -1/9 (exact).
J3  T52 part 2: a unit read alone, r = relu(h - 1/2), h uniform on {0, 1/4, 1, 2}: gamma = 0 and
    delta = target mass of {h > 1/2} = 3/13 + ... (exact formula printed), and delta = 0 for r = relu(h + 1) (b > 0 is
    outside part 2; there the rest alone keeps r on).
J4  T52 part 1, random wide layer (illustration, numbers only in the JSON): 400 units over 60 Gaussian inputs, 120
    readers with N(0, 1/400) weights, no bias. delta in [0, 1] for every unit; gamma > 0 for about half of the units.
Prediction (written before the run): J1 equal for every instance, same a for equal n_bad; J2 1, 0, -1/9; J3 gamma 0 and
delta as printed, 0 for b > 0; J4 every delta in [0, 1], share of gamma > 0 between 1/4 and 3/4.
"""
import itertools
import json
import os
import sys
from fractions import Fraction as Fr

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from t43_clean_hard import build, colored_graph, detector  # noqa: E402

out = {}

# ---------------- J1: Lemma T51 ----------------
def jobmix_check(H, kappa, m):
    I = build(colored_graph(H, kappa, m), kappa, m)
    D = detector(kappa, m)
    Z = I["Z"]
    P = {x: w / Z for x, w in zip(I["pts"], I["w"])}
    # targets of layer-1 units under nu = P, and the silent part
    units = list(range(len(D["W"])))
    acts = {x: [max(sum(a * b for a, b in zip(row, x)) - D["r"], Fr(0)) for row in D["W"]] for x in P}
    targets = {}
    for u in units:
        mass = sum(P[x] * acts[x][u] for x in P)
        if mass > 0:
            targets[u] = {x: P[x] * acts[x][u] / mass for x in P if acts[x][u] > 0}
    silent = [x for x in P if all(h == 0 for h in acts[x])]
    s_mass = sum(P[x] for x in silent)
    mu0 = {x: P[x] / s_mass for x in silent}
    n_bad = len(I["bad"])
    a = {u: Fr(1) / Z for u in targets}
    a0 = I["M"] * n_bad / Z
    Pa = {}
    for u, t in targets.items():
        for x, q in t.items():
            Pa[x] = Pa.get(x, 0) + a[u] * q
    for x, q in mu0.items():
        Pa[x] = Pa.get(x, 0) + a0 * q
    ok = Pa == P and sum(a.values()) + a0 == 1 and len(targets) == kappa * m
    return ok, n_bad, (Z, a0), frozenset(P.items())

J1 = []
cases = [({frozenset((1, 2))}, 2, 2), ({frozenset((1, 2)), frozenset((2, 3))}, 2, 3),
         ({frozenset((1, 3)), frozenset((2, 3))}, 2, 3), ({frozenset((1, 2)), frozenset((1, 3)), frozenset((2, 3))}, 3, 3)]
res = [jobmix_check(*c) for c in cases]
for c, r in zip(cases, res):
    J1.append(dict(H=sorted(tuple(sorted(e)) for e in c[0]), kappa=c[1], m=c[2], equal=r[0], n_bad=r[1]))
same_a_diff_P = res[1][1] == res[2][1] and res[1][2] == res[2][2] and res[1][3] != res[2][3]
out["J1"] = dict(cases=J1, all_equal=all(r[0] for r in res), same_a_different_P=same_a_diff_P)
print("J1", out["J1"])

# ---------------- D49 indices on a finite law (exact) ----------------
def indices_exact(h_u, R, b, W_ru, law):
    """one reader r = relu(W_ru h_u + R), R = rest incl. bias b; R' = R - b. law: list of (prob, h_u, R)."""
    E_h = sum(p * hu for p, hu, _ in law)
    mu = [(p * hu / E_h, hu, Rr) for p, hu, Rr in law if hu > 0]
    T = abs(W_ru)
    on = lambda z: 1 if z > 0 else 0
    delta = sum(q * W_ru * (on(W_ru * hu + Rr) - on(Rr)) for q, hu, Rr in mu) / T
    gamma = sum(q * (1 if W_ru > 0 else -1) * (Rr - b) for q, hu, Rr in mu)
    S = lambda hu, Rr: W_ru * on(W_ru * hu + Rr)
    lam = (sum(q * S(hu, Rr) for q, hu, Rr in mu) - sum(p * S(hu, Rr) for p, hu, Rr in law)) / T
    lam_ind = (sum(q * p2 * S(hu, R2) for q, hu, _ in mu for p2, _, R2 in law)
               - sum(p * p2 * S(hu, R2) for p, hu, _ in law for p2, _, R2 in law)) / T
    return dict(delta=delta, gamma=gamma, lam=lam, xi=lam - lam_ind)

# J2: OR reader. points (1,0), (0,1), (0,0): h_1 = (1, 0, 0), rest = h_2 = (0, 1, 0)
third = Fr(1, 3)
J2 = indices_exact(None, None, Fr(0), Fr(1), [(third, Fr(1), Fr(0)), (third, Fr(0), Fr(1)), (third, Fr(0), Fr(0))])
out["J2"] = {k: str(v) for k, v in J2.items()}
print("J2", out["J2"])

# J3: read alone, bias -1/2 and +1
law_h = [Fr(0), Fr(1, 4), Fr(1), Fr(2)]
J3a = indices_exact(None, None, Fr(-1, 2), Fr(1), [(Fr(1, 4), h, Fr(-1, 2)) for h in law_h])
J3b = indices_exact(None, None, Fr(1), Fr(1), [(Fr(1, 4), h, Fr(1)) for h in law_h])
E_h = sum(law_h) / 4
pred = sum(Fr(1, 4) * h / E_h for h in law_h if h > Fr(1, 2))
out["J3"] = dict(bias_neg={k: str(v) for k, v in J3a.items()}, predicted_delta=str(pred),
                 bias_pos={k: str(v) for k, v in J3b.items()})
print("J3", out["J3"])

# ---------------- J4: random wide layer (illustration) ----------------
rng = np.random.default_rng(0)
n_in, n, n_r, N = 60, 400, 120, 20000
X = rng.standard_normal((N, n_in))
W1 = rng.standard_normal((n, n_in)) / np.sqrt(n_in)
H = np.maximum(X @ W1.T, 0)
W2 = rng.standard_normal((n_r, n)) / np.sqrt(n)
Z2 = H @ W2.T
deltas, gammas = [], []
for u in range(60):
    hu = H[:, u]
    mu = hu / hu.sum()
    Rr = Z2 - np.outer(hu, W2[:, u])  # rest, no bias
    dec = (Z2 > 0).astype(float) - (Rr > 0).astype(float)
    T = np.abs(W2[:, u]).sum()
    deltas.append(float(mu @ (dec @ W2[:, u]) / T))
    gammas.append(float(mu @ (Rr @ np.sign(W2[:, u]))))
out["J4"] = dict(delta_min=min(deltas), delta_max=max(deltas), delta_median=float(np.median(deltas)),
                 gamma_pos_share=float(np.mean(np.array(gammas) > 0)))
print("J4", out["J4"])

ok = (out["J1"]["all_equal"] and out["J1"]["same_a_different_P"]
      and J2["delta"] == 1 and J2["gamma"] == 0 and J2["xi"] == Fr(-1, 9)
      and J3a["gamma"] == 0 and J3a["delta"] == pred and J3b["delta"] == 0
      and 0 <= out["J4"]["delta_min"] and out["J4"]["delta_max"] <= 1
      and 0.25 <= out["J4"]["gamma_pos_share"] <= 0.75)
out["ALL_OK"] = bool(ok)
print("ALL OK" if ok else "MISMATCH")
json.dump(out, open(os.path.join(os.path.dirname(__file__), "t51_jobmix_listen.json"), "w"), indent=1, default=str)
