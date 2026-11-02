# serves: T45
# serves: T44
"""Checks of Remark A.5.3, Theorem T44 and Theorem T45 (COARSE-THEORY.md A.5 and B.Q).

E1  Sign structure of every reader of the Part B networks, in exact rationals: which readers read two units with
    opposite signs (eps_V = 0 when there are none; Remark A.5.3).
E2  Broken lines (teachers of T22 and T23): at x*, the mix point with the largest v^T x, every unit is on (exact), and
    some reader reads two of them with opposite signs (so eps_V > 0).
E3  Detector network D40 (kappa 2, m 3), L = 2, on its own domain: the mass of h_f over each layer-1 region, by
    quadrature; equal masses give g_f = 1 - 1/(kappa m) = 5/6.
E4  Theorem T45: the example of part (3) and the T20 reader relu(x - 1) - relu(-x - 1) stay margin-clean under small
    random changes of all weights, and their sampled eps is 0.
E5  Theorem T44: on the box [-1, 1]^2, the T32 network has a symmetry that moves units; after a small random change of
    its weights no symmetry moves a unit; random networks have none.
E6  Sampled eps of the Part B networks on a ball that holds the mix (numbers for the record, not for the paper).
E7  Referee report 21, problem 1: eps_V on raw weights moves with a reader's scale c; on the reduced form it does not.
E9  Comment (h) of D42: the ambulance and packing circuits, and the teachers, under D42's V and under the candidate.
E8  Comment (e) of D42: eps_V = 0 for relu(x), relu(-x) read with +1, -1; > 0 once both biases get delta > 0.
Prediction (written before the run): E1 opposite pairs only for the broken lines; E2 all on, opposite pair found;
E3 masses equal to quadrature accuracy, g_f = 5/6; E4 every perturbed copy margin-clean, eps = (0, 0, 0); E5 T32
moves units, its perturbed copies and the random networks do not; E6 eps_V > 0 only for the broken lines;
E7 raw eps_V falls toward 0 as c grows, reduced eps_V is the same for every c; E8 0, then > 0;
E9 D42's V > 0 on the ambulance (the defect); the candidate gives the ambulance small V and B = 0, the packing
circuit B > 0; the T22, T23 teachers keep V > 0; T20, T32 stay at 0.
"""
import itertools
import json
import os
import sys
from fractions import Fraction as Fr

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from clean_eps import b_last, eps, eps_v_reader, sample_ball, sample_box  # noqa: E402
from t22_reduction import build, circle_pt  # noqa: E402

rng = np.random.default_rng(0)
out = {}


def broken_line(pts, logit_rows):
    """Lemma T22a, as in ta5_principles.py: units (v, -s_(j)) and one reader row per logit difference."""
    n, d = len(pts), len(pts[0])
    for tau in range(1, (d - 1) * n * n + 2):
        v = [Fr(tau) ** l for l in range(d)]
        s = [sum(a * b for a, b in zip(v, p)) for p in pts]
        if len(set(s)) == n:
            break
    order = sorted(range(n), key=lambda i: s[i])
    ss = [s[i] for i in order]
    units = [(tuple(v), -ss[j]) for j in range(n - 1)]
    readers = []
    for row in logit_rows:
        tt = [row[i] for i in order]
        mu = [Fr(0)] + [(tt[j + 1] - tt[j]) / (ss[j + 1] - ss[j]) for j in range(n - 1)]
        readers.append([mu[j + 1] - mu[j] for j in range(n - 1)])
    return units, readers


def rot_z(p):
    return (-p[1], p[0]) + tuple(p[2:])


def orbit(p):
    o = [tuple(Fr(c) for c in p)]
    for _ in range(3):
        o.append(rot_z(o[-1]))
    return o


nets = {}
nets["T20"] = dict(units=[((Fr(1),), Fr(-1)), ((Fr(-1),), Fr(-1))], readers=[[Fr(1), Fr(1)]],
                   pts=[(Fr(x),) for x in range(-3, 4)])
I22 = build({frozenset(((0, 1), (1, 2)))})
u, r = broken_line(I22["pts"], [I22["lab"]])
nets["T22 teacher"] = dict(units=u, readers=r, pts=I22["pts"], broken=True)
p23 = [(Fr(i - 3), Fr(0)) for i in range(1, 6)] + [(Fr(1), Fr(1))]
u, r = broken_line(p23, [[Fr(c) for c in (2, 1, 0, -2, -4, 0)], [Fr(c) for c in (0, 0, 0, 0, 0, 1)]])
nets["T23 teacher"] = dict(units=u, readers=r, pts=p23, broken=True)
dirs4 = [(1, 0), (0, 1), (-1, 0), (0, -1)]
nets["T32"] = dict(units=[(tuple(Fr(c) for c in w), Fr(-1)) for w in dirs4], readers=[[Fr(1)] * 4],
                   pts=[(Fr(a), Fr(b)) for a, b in [(2, 0), (0, 2), (-2, 0), (0, -2), (2, 2), (-2, 2), (-2, -2), (2, -2)]])
wA = [(1, 0, Fr(1, 2)), (0, 1, Fr(1, 2)), (-1, 0, Fr(1, 2)), (0, -1, Fr(1, 2))]
wB = [(1, 1, Fr(-1, 2)), (-1, 1, Fr(-1, 2)), (-1, -1, Fr(-1, 2)), (1, -1, Fr(-1, 2))]
nets["T33"] = dict(units=[(tuple(Fr(c) for c in w), Fr(-1)) for w in wA], readers=[[Fr(1)] * 4],
                   pts=orbit((-1, 2, -2)) + orbit((2, 1, -1)) + orbit((1, 0, 2)))
nets["T34"] = dict(units=[(tuple(Fr(c) for c in w), Fr(-1)) for w in wA + wB], readers=[[Fr(1)] * 8],
                   pts=orbit((2, 0, 2)) + orbit((-1, 2, 0)) + orbit((1, -1, 0)))
kappa, m = 2, 3
xs = [circle_pt(i) for i in range(1, m + 1)]
eta = 1 - max(a[0] * b[0] + a[1] * b[1] for a, b in itertools.permutations(xs, 2))
rr = 1 - eta / 8
u40 = []
for j in range(kappa):
    for i in range(m):
        w = [Fr(0)] * (2 * kappa)
        w[2 * j], w[2 * j + 1] = xs[i]
        u40.append((tuple(w), -rr))
I40 = build({frozenset(((0, a), (1, b))) for a, b in [(1, 2), (2, 1), (2, 3), (3, 2)]}, kappa=kappa, per_color=m)
nets["D40 L=1"] = dict(units=u40, readers=[[Fr(1)] * len(u40)], pts=I40["pts"], sample=False)

# E1 and E2
e1 = {}
for name, N in nets.items():
    opp = [(ri, a, b) for ri, rw in enumerate(N["readers"]) for a, b in itertools.combinations(range(len(rw)), 2)
           if rw[a] * rw[b] < 0]
    e1[name] = len(opp)
    line = f"E1 {name}: opposite-sign pairs {len(opp)}"
    if N.get("broken"):
        U = N["units"]
        v = U[0][0]
        xstar = max(N["pts"], key=lambda p: sum(a * c for a, c in zip(v, p)))
        z = [sum(a * c for a, c in zip(w, xstar)) + b for w, b in U]
        allon = all(t > 0 for t in z)
        opp_on = [o for o in opp if z[o[1]] > 0 and z[o[2]] > 0]
        line += f"; E2 all units on at x*: {allon}, min z {min(z)}; opposite pairs on at x*: {len(opp_on)}"
        out.setdefault("E2", {})[name] = dict(all_on=allon, min_z=str(min(z)), opposite_on=len(opp_on))
    print(line)
out["E1"] = e1

# E3: D40, L = 2. Mass of h_f = d_(j,i) over region (j,i) on X = {|y_1| + |y_2| <= 1} (blocks y_1, y_2 in R^2):
# integral over block j of (<x~_i, y> - r)_+ times the area pi (1 - |y|)^2 of the other block's disc.
rf = float(rr)
masses = []
nr, nt = 4000, 4000
rad = np.linspace(rf, 1.0, nr + 1)
rad = (rad[:-1] + rad[1:]) / 2
dr = (1.0 - rf) / nr
for i in range(m):
    a = np.array([float(c) for c in xs[i]])
    th0 = np.arctan2(a[1], a[0])
    halfw = np.arccos(rf)
    th = np.linspace(th0 - halfw, th0 + halfw, nt + 1)
    th = (th[:-1] + th[1:]) / 2
    dth = 2 * halfw / nt
    R_, T_ = np.meshgrid(rad, th, indexing="ij")
    val = np.maximum(R_ * np.cos(T_ - th0) - rf, 0.0) * np.pi * (1 - R_) ** 2 * R_
    masses.append(float(val.sum() * dr * dth))
masses = masses * kappa  # block permutations carry block 1 onto block 2
share = max(masses) / sum(masses)
gf = 1 - share
print(f"E3 D40 L=2: region masses {['%.6e' % x for x in masses[:m]]} (x {kappa} blocks); "
      f"spread {(max(masses) - min(masses)) / max(masses):.2e}; g_f = 1 - largest share = {gf:.6f} (5/6 = {5/6:.6f})")
out["E3"] = dict(masses=masses, g_f=gf, target=5 / 6)


# E4: Theorem T45 part (2), (3)
def margin_ok(layers, W_out, x):
    from clean_eps import forward, readers
    zs, hs = forward(layers, x)
    ok = True
    for l in range(len(layers)):
        R = readers(layers, W_out, l)
        for rw in R:
            for a, b in itertools.combinations(range(len(rw)), 2):
                if rw[a] * rw[b] < 0 and np.any((zs[l][:, a] >= 0) & (zs[l][:, b] >= 0)):
                    ok = False
        if l >= 1:
            for uu in range(zs[l].shape[1]):
                on = zs[l][:, uu] >= 0
                if not any(layers[l][0][uu, v] > 0 and np.all(zs[l - 1][on, v] > 0) for v in range(zs[l - 1].shape[1])):
                    ok = False
        if l <= len(layers) - 2:
            for uu in range(zs[l].shape[1]):
                on = zs[l][:, uu] >= 0
                if not any(layers[l + 1][0][r_, uu] > 0 and np.all(zs[l + 1][on, r_] > 0)
                           for r_ in range(zs[l + 1].shape[1])):
                    ok = False
    return ok


d, widths, n_y = 3, (3, 2), 3
s_ = 3.0
t_ = 1 / (2 * s_ + 2)
ex = [(np.array([[t_ / (u + 1)] * d for u in range(widths[0])]), np.ones(widths[0])),
      (np.array([[(u + 1) + (v + 1) for v in range(widths[0])] for u in range(widths[1])], float), np.ones(widths[1]))]
Wo = np.array([[c + 1.0] * widths[1] for c in range(n_y)])
xb = sample_box([-1] * d, [1] * d, 20000, rng)
e4 = []
for trial in range(20):
    lay = [(W + 1e-2 * rng.standard_normal(W.shape), b + 1e-2 * rng.standard_normal(b.shape)) for W, b in ex]
    Wp = Wo + 1e-2 * rng.standard_normal(Wo.shape)
    E = eps(lay, Wp, xb)
    e4.append(dict(margin=margin_ok(lay, Wp, xb), eps=[E["eps_V"], E["eps_G"], E["eps_B"]]))
x20 = np.linspace(-3, 3, 20001)[:, None]
e4b = []
for trial in range(20):
    lay = [(np.array([[1.0], [-1.0]]) + 1e-2 * rng.standard_normal((2, 1)), np.array([-1.0, -1.0]) + 1e-2 * rng.standard_normal(2))]
    Wp = np.array([[1.0, -1.0], [0.0, 0.0]]) + 1e-2 * rng.standard_normal((2, 2))
    E = eps(lay, Wp, x20)
    e4b.append(dict(margin=margin_ok(lay, Wp, x20), eps=[E["eps_V"], E["eps_G"], E["eps_B"]]))
print(f"E4 T45 example (3), 20 perturbed copies: all margin-clean {all(r_['margin'] for r_ in e4)}, "
      f"max eps {max(max(r_['eps']) for r_ in e4):.2e}")
print(f"E4 T20 reader relu(x-1) - relu(-x-1), 20 perturbed copies: all margin-clean {all(r_['margin'] for r_ in e4b)}, "
      f"max eps {max(max(r_['eps']) for r_ in e4b):.2e}")
out["E4"] = dict(example3=e4, t20=e4b)


# E5: Theorem T44 on the box [-1, 1]^2, one hidden layer
def moves_unit(W, b, Wout, cout, tol=1e-9):
    A_list = []
    for perm in itertools.permutations(range(2)):
        for sg in itertools.product([1, -1], repeat=2):
            A = np.zeros((2, 2))
            for i, j in enumerate(perm):
                A[i, j] = sg[i]
            A_list.append(A)
    rows = np.hstack([W, b[:, None]])
    nrm = np.linalg.norm(rows, axis=1, keepdims=True)
    Rn = rows / nrm
    Wo = Wout * nrm.T  # move the scales into the readout (reduced form)
    n = W.shape[0]
    for A in A_list:
        moved = np.hstack([W @ A, b[:, None]]) / nrm
        for pi in itertools.permutations(range(n)):
            if all(pi[u] == u for u in range(n)):
                continue
            if not np.allclose(moved, Rn[list(pi)], atol=tol):
                continue
            for sig in itertools.permutations(range(Wout.shape[0])):
                if np.allclose(Wo[:, list(pi)], Wo[list(sig)], atol=tol) and np.allclose(cout, cout[list(sig)], atol=tol):
                    return True
    return False


W32 = np.array(dirs4, float)
b32 = -np.ones(4)
Wout32 = np.vstack([np.ones(4), np.zeros(4)])
c32 = np.zeros(2)
sym32 = moves_unit(W32, b32, Wout32, c32)
pert = [moves_unit(W32 + 1e-3 * rng.standard_normal(W32.shape), b32 + 1e-3 * rng.standard_normal(4),
                   Wout32 + 1e-3 * rng.standard_normal(Wout32.shape), c32) for _ in range(50)]
rand = [moves_unit(rng.standard_normal((4, 2)), rng.standard_normal(4), rng.standard_normal((2, 4)), rng.standard_normal(2))
        for _ in range(50)]
print(f"E5 T32 on the box: a symmetry moves a unit: {sym32}; perturbed copies with such a symmetry: {sum(pert)}/50; "
      f"random networks with such a symmetry: {sum(rand)}/50")
out["E5"] = dict(t32=sym32, perturbed=sum(pert), random=sum(rand))

# E6: sampled eps on a ball that holds the mix (one hidden layer, logits (reader, 0))
e6 = {}
for name, N in nets.items():
    if N.get("sample") is False:
        continue
    W = np.array([[float(c) for c in w] for w, _ in N["units"]])
    b = np.array([float(bb) for _, bb in N["units"]])
    Wout = np.vstack([np.array([[float(c) for c in rw] for rw in N["readers"]]), np.zeros((1, W.shape[0]))])
    P = np.array([[float(c) for c in p] for p in N["pts"]])
    cen = P.mean(axis=0)
    rad_ = 1.05 * np.max(np.linalg.norm(P - cen, axis=1))
    E = eps([(W, b)], Wout, sample_ball(cen, rad_, 200000, rng))
    e6[name] = [E["eps_V"], E["eps_G"], E["eps_B"]]
    print(f"E6 {name}: sampled eps (V, G, B) = ({E['eps_V']:.3g}, {E['eps_G']:.3g}, {E['eps_B']:.3g})")
out["E6"] = e6
# E7 (referee report 21, problem 1): scaling a reader changes eps computed on raw weights, not on the reduced form
x1 = np.linspace(-1, 1, 40001)[:, None]
e7 = []
for c in [1.0, 10.0, 100.0, 1000.0]:
    lay = [(np.array([[1.0], [1.0]]), np.array([0.0, 0.5])), (np.array([[1.0, -1.0], [c, 0.0]]), np.array([1.0, 0.0]))]
    Wo_ = np.array([[1.0, 1.0 / c], [0.0, 0.0]])
    raw = eps(lay, Wo_, x1, reduce=False)["eps_V"]
    red = eps(lay, Wo_, x1)["eps_V"]
    e7.append(dict(c=c, raw=raw, reduced=red))
    print(f"E7 c = {c:g}: eps_V on raw weights {raw:.4f}, on the reduced form {red:.4f}")
out["E7"] = e7
# E8 (comment (e) of D42): relu(x), relu(-x) read with +1, -1 on [-1, 1]; eps_V = 0, and > 0 after adding delta to both biases
e8 = []
for dl in [0.0, 1e-3, 1e-2]:
    lay = [(np.array([[1.0], [-1.0]]), np.array([dl, dl]))]
    e8.append(dict(delta=dl, eps_V=eps(lay, np.array([[1.0, -1.0], [0.0, 0.0]]), x1)["eps_V"]))
    print(f"E8 delta = {dl:g}: eps_V {e8[-1]['eps_V']:.2e}")
out["E8"] = e8
# E9 (comment (h) of D42): the essay's ambulance circuit and an arbitrary-packing circuit, under D42's V and under the
# candidate (V on each reader's own target, B extended to the last layer). X = [0, 1]^2, one hidden layer.
x2 = sample_box([0, 0], [1, 1], 200000, rng)
lay_amb = [(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([-0.3, -0.3]))]  # u = car body, v = yellow sign
Wo_amb = np.array([[1.0, -2.0], [1.0, 1.0], [0.0, 0.0]])  # ambulance = u - 2v, taxi = u + v, other = 0
lay_pack = [(np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]), np.array([-1.0, -0.6, -0.6]))]  # u packs A and B; a, b
Wo_pack = np.array([[1.0, 0.0, -3.0], [1.0, -3.0, 0.0], [0.0, 0.0, 0.0]])  # class A = u - 3b, class B = u - 3a
e9 = {}
for name, (lay, Wo_) in {"ambulance": (lay_amb, Wo_amb), "packing": (lay_pack, Wo_pack)}.items():
    E = eps(lay, Wo_, x2)
    e9[name] = dict(V=E["eps_V"], V_reader=eps_v_reader(lay, Wo_, x2), B_last=b_last(lay, Wo_, x2))
    print(f"E9 {name}: D42 eps_V {e9[name]['V']:.3f}; candidate: V on readers' own targets {e9[name]['V_reader']:.3f}, "
          f"B at the last layer {e9[name]['B_last']:.3f}")
for name in ["T22 teacher", "T23 teacher", "T20", "T32"]:
    N = nets[name]
    W = np.array([[float(c) for c in w] for w, _ in N["units"]])
    bb = np.array([float(c) for _, c in N["units"]])
    Wout = np.vstack([np.array([[float(c) for c in rw] for rw in N["readers"]]), np.zeros((1, W.shape[0]))])
    P_ = np.array([[float(c) for c in p] for p in N["pts"]])
    cen = P_.mean(axis=0)
    xs_ = sample_ball(cen, 1.05 * np.max(np.linalg.norm(P_ - cen, axis=1)), 200000, rng)
    e9[name] = dict(V_reader=eps_v_reader([(W, bb)], Wout, xs_), B_last=b_last([(W, bb)], Wout, xs_))
    print(f"E9 {name}: candidate V on readers' own targets {e9[name]['V_reader']:.3f}, B at the last layer "
          f"{e9[name]['B_last']:.3f}")
out["E9"] = e9
json.dump(out, open(os.path.join(os.path.dirname(__file__), "t45_clean.json"), "w"), indent=1, default=str)
