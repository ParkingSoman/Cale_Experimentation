# serves: T61
"""Proposition T61 (COARSE-THEORY.md B.W): the build relu(-20 z_A - 40 z_C - 45/2) + relu(-40 z_B - 20 z_D - 45/2) - 18
equals N's gap on the mix of T43 (kappa = 2, m = 2, H one edge), in exact fractions."""
import itertools
import json
import os
from fractions import Fraction as Fr


def circle_pt(i):
    return (Fr(1 - i * i, 1 + i * i), Fr(2 * i, 1 + i * i))


xs = [circle_pt(1), circle_pt(2)]
eta = 1 - max(a[0] * b[0] + a[1] * b[1] for a, b in itertools.permutations(xs, 2))
r = 1 - eta / 8
c = 2 / (1 - r)
units = {"A": (0, 0), "B": (0, 1), "C": (1, 0), "D": (1, 1)}   # (block, direction index)


def vertex(u):
    j, i = units[u]
    x = [Fr(0)] * 4
    x[2 * j], x[2 * j + 1] = xs[i]
    return x


def z(u, x):
    j, i = units[u]
    return xs[i][0] * x[2 * j] + xs[i][1] * x[2 * j + 1] - r


def relu(t):
    return t if t > 0 else Fr(0)


bad = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D")]
pts = [(u, vertex(u)) for u in "ABCD"] + [("mid" + a + b, [(p + q) / 2 for p, q in zip(vertex(a), vertex(b))]) for a, b in bad]
rows, ok = [], True
for name, x in pts:
    gapN = c * sum(relu(z(u, x)) for u in "ABCD")
    l1 = -20 * z("A", x) - 40 * z("C", x) - Fr(45, 2)
    l2 = -40 * z("B", x) - 20 * z("D", x) - Fr(45, 2)
    gapB = relu(l1) + relu(l2) - 18
    ok &= gapB == gapN
    rows.append(dict(point=name, gapN=str(gapN), pre1=str(l1), pre2=str(l2), gapB=str(gapB)))
    print(f"{name:6} N {str(gapN):>3}  pre ({str(l1):>4}, {str(l2):>4})  build {gapB}")
print("r =", r, "c =", c, "ALL_OK", ok)
with open(os.path.join(os.path.dirname(__file__), "tr_acbd.json"), "w") as f:
    json.dump(dict(rows=rows, ALL_OK=ok), f, indent=1)
