# serves: T15
"""Exact proof computation for Theorem T15 (COARSE-THEORY.md B.6).

Chain: depth 1, N_1 = 2, one position, z = a in R^2 (K = I), gate 1[s^T z > 0], logit difference
tau(x) = 1[s^T z > 0] alpha^T z. Law: uniform on 4 points z_1..z_4 of Z^2. Width k = r = 1.
Self-contained width-1 builds: tau_B = (beta^T z) 1[mu^T z > 0] + b with beta a real multiple of mu (proof of T15).
D2 = 1/2 E (tau_B - tau)^2.

Step 1 (search, floating point): small integer instances with 0 < min over builds < T11 value.
Step 2 (proof, exact rationals, fractions.Fraction): for the chosen instance, the T11 value and the minimum over the 8
closed sectors x 2 signs of the convex problem over (beta in the closed cone, b), each solved exactly by the stationary
point or its faces. Prediction (written before the run): Step 2 confirms 0 < min < T11 value exactly for the instance
that Step 1 picks, and the minimising build has a pattern or a direction different from the forced read s.
"""
import itertools
import json
import math
import random
from fractions import Fraction as Fr


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def centred_sq(vec):
    m = sum(vec) / len(vec)
    return sum((v - m) ** 2 for v in vec)


def fit_1d(f, tau, sign_constraint=None):
    """min over theta (theta >= 0 if sign_constraint) and b of 1/2 * 1/4 * |theta f + b - tau|^2. Returns (value, theta)."""
    n = len(tau)
    fm, tm = sum(f) / n, sum(tau) / n
    fc = [x - fm for x in f]
    tc = [x - tm for x in tau]
    ff = dot(fc, fc)
    theta = dot(fc, tc) / ff if ff != 0 else 0 * tm
    if sign_constraint and theta < 0:
        theta = 0 * tm
    res = [tc[i] - theta * fc[i] for i in range(n)]
    return dot(res, res) / (2 * n), theta


def fit_cone(X, r1, r2, tau):
    """min over beta = t1 r1 + t2 r2 (t >= 0) and b of 1/2 * 1/n * |X beta + b - tau|^2 (exact if inputs exact)."""
    n = len(tau)
    f1 = [dot(row, r1) for row in X]
    f2 = [dot(row, r2) for row in X]
    best = None
    # faces: rays and the origin
    for f in (f1, f2):
        v, th = fit_1d(f, tau, sign_constraint=True)
        if best is None or v < best[0]:
            best = (v, ("ray", th))
    # interior stationary point
    m1, m2, mt = sum(f1) / n, sum(f2) / n, sum(tau) / n
    a = [x - m1 for x in f1]; c = [x - m2 for x in f2]; t = [x - mt for x in tau]
    g11, g12, g22 = dot(a, a), dot(a, c), dot(c, c)
    det = g11 * g22 - g12 * g12
    if det != 0:
        h1, h2 = dot(a, t), dot(c, t)
        t1 = (g22 * h1 - g12 * h2) / det
        t2 = (g11 * h2 - g12 * h1) / det
        if t1 >= 0 and t2 >= 0:
            res = [t[i] - t1 * a[i] - t2 * c[i] for i in range(n)]
            v = dot(res, res) / (2 * n)
            if v < best[0]:
                best = (v, ("interior", (t1, t2)))
    return best


def sectors(Z):
    dirs = []
    for z in Z:
        rz = (-z[1], z[0])
        dirs += [rz, (-rz[0], -rz[1])]
    dirs.sort(key=lambda d: math.atan2(float(d[1]), float(d[0])))
    out = []
    for j in range(len(dirs)):
        r1, r2 = dirs[j], dirs[(j + 1) % len(dirs)]
        mid = (r1[0] + r2[0], r1[1] + r2[1])
        A = tuple(i for i, z in enumerate(Z) if dot(mid, z) > 0)
        out.append((r1, r2, A))
    return out


def analyse(Z, s, alpha):
    tau = [dot(alpha, z) if dot(s, z) > 0 else 0 * dot(alpha, z) for z in Z]
    t11, _ = fit_1d([max(dot(s, z), 0 * dot(s, z)) for z in Z], tau)
    best = None
    for r1, r2, A in sectors(Z):
        X = [z if i in A else (0 * z[0], 0 * z[1]) for i, z in enumerate(Z)]
        for sg in (1, -1):
            v, how = fit_cone(X, (sg * r1[0], sg * r1[1]), (sg * r2[0], sg * r2[1]), tau)
            if best is None or v < best[0]:
                best = (v, A, sg, (r1, r2), how)
    var0 = centred_sq(tau) / (2 * len(tau))
    return t11, best, var0


def general_position(Z):
    for z1, z2 in itertools.combinations(Z, 2):
        if z1[0] * z2[1] - z1[1] * z2[0] == 0:
            return False
    return all(z != (0, 0) for z in Z)


random.seed(15)
cands = []
for _ in range(20000):
    Z = [(random.randint(-3, 3), random.randint(-3, 3)) for _ in range(4)]
    if not general_position(Z):
        continue
    s = (random.randint(-2, 2), random.randint(-2, 2))
    al = (random.randint(-2, 2), random.randint(-2, 2))
    if s == (0, 0) or al == (0, 0):
        continue
    if not any(dot(s, z) > 0 for z in Z) or all(dot(s, z) > 0 for z in Z):
        continue
    t11, best, var0 = analyse(Z, s, al)
    if t11 > 0 and best[0] > 0.05 * t11 and best[0] < 0.7 * t11:
        cands.append((sum(abs(c) for z in Z for c in z) + sum(map(abs, s)) + sum(map(abs, al)), Z, s, al, t11, best[0]))
cands.sort(key=lambda c: c[0])
_, Z, s, al, t11f, inff = cands[0]
print("search: candidates", len(cands), "chosen", Z, s, al, "float t11", t11f, "float inf", inff)

# exact verification
Zx = [(Fr(z[0]), Fr(z[1])) for z in Z]
sx, ax = (Fr(s[0]), Fr(s[1])), (Fr(al[0]), Fr(al[1]))
t11, best, var0 = analyse(Zx, sx, ax)
v, A, sg, (r1, r2), how = best
out = {"points z_i": Z, "s (gate read)": s, "alpha (teacher's linear read)": al,
       "teacher tau on the points": [str(dot(ax, z) if dot(sx, z) > 0 else 0) for z in Zx],
       "T11 value (exact)": str(t11), "min over self-contained builds (exact)": str(v),
       "k = 0 value 1/2 Var(tau) (exact)": str(var0),
       "minimiser: pattern A (indices with gate on)": list(A), "sign": sg,
       "sector rays": [list(map(str, r1)), list(map(str, r2))], "minimiser form": [how[0], str(how[1])],
       "0 < min < T11": bool(0 < v < t11),
       "forced-read pattern": [i for i, z in enumerate(Zx) if dot(sx, z) > 0]}
print(json.dumps(out, indent=1))
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
