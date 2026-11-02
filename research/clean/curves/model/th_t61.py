# serves: TH2
"""Proposition TH2 part 1 (COARSE-THEORY.md B.H) against Proposition T61. T61's grouping {A, C}, {B, D} has V_b = 0 but
the step-by-step zoom through {A, C} ends above V_b. TH2 part 1 would contradict that if T61's instance had a tree map
meeting TH1's premises. This script tries every region for every unit (all subsets of the 8 points of P) and checks,
in exact fractions: district error 0, shelter error 0, and the two groups on disjoint areas (DH3 item 2). Expected:
no assignment passes, because each unit's pre-activation varies outside every region that is not all of P."""
import itertools
import json
import os
from fractions import Fraction as Fr


def circle_pt(i):
    return (Fr(1 - i * i, 1 + i * i), Fr(2 * i, 1 + i * i))


xs = [circle_pt(1), circle_pt(2)]
eta = 1 - max(a[0] * b[0] + a[1] * b[1] for a, b in itertools.permutations(xs, 2))
r = 1 - eta / 8
units = {"A": (0, 0), "B": (0, 1), "C": (1, 0), "D": (1, 1)}


def vertex(u):
    j, i = units[u]
    x = [Fr(0)] * 4
    x[2 * j], x[2 * j + 1] = xs[i]
    return x


def z(u, x):
    j, i = units[u]
    return xs[i][0] * x[2 * j] + xs[i][1] * x[2 * j + 1] - r


bad = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D")]
pts = [vertex(u) for u in "ABCD"] + [[(p + q) / 2 for p, q in zip(vertex(a), vertex(b))] for a, b in bad]
n = len(pts)
zv = {u: [z(u, x) for x in pts] for u in "ABCD"}
support = {u: frozenset(k for k in range(n) if zv[u][k] > 0) for u in "ABCD"}


def ok_region(u, R):
    if not support[u] <= R:
        return False                                    # district error > 0
    outside = [zv[u][k] for k in range(n) if k not in R]
    return len(set(outside)) <= 1                       # shelter error 0 iff constant outside


subsets = [frozenset(c) for m in range(n + 1) for c in itertools.combinations(range(n), m)]
good = {u: [R for R in subsets if ok_region(u, R)] for u in "ABCD"}
passes = 0
for RA, RC, RB, RD in itertools.product(good["A"], good["C"], good["B"], good["D"]):
    regs = [RA, RC, RB, RD]
    if any(not (a == b or not (a & b)) for a, b in itertools.combinations(regs, 2)):
        continue                                        # DM2: one layer's regions equal or disjoint
    if any(a & b for a in (RA, RC) for b in (RB, RD)):
        continue                                        # DH3 item 2: the two groups on disjoint areas
    passes += 1
res = dict(regions_with_district_and_shelter_0={u: [sorted(R) for R in good[u]] for u in "ABCD"},
           supports={u: sorted(support[u]) for u in "ABCD"}, passing_maps=passes, ALL_OK=passes == 0)
for u in "ABCD":
    print(u, "support", sorted(support[u]), "regions with district and shelter error 0:", [sorted(R) for R in good[u]])
print("tree maps meeting TH1's premises for {A,C},{B,D}:", passes, "ALL_OK", passes == 0)
with open(os.path.join(os.path.dirname(__file__), "th_t61.json"), "w") as f:
    json.dump(res, f, indent=1)
