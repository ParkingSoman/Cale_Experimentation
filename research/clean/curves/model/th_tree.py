# serves: TH2
"""Theorem TH1 and Proposition TH2 part 1 (COARSE-THEORY.md B.H) on a small gated network, in exact fractions.

Network (2 hidden layers, 2 logits). Layer 1: a car gate district (units K1, K1', K2, K2', I3, I4; area = taxi and
ambulance photos) and a dog district (Dg). Layer 2: taxi units T1, T2 and judgment J1 (area taxi), ambulance units A1,
A2 and judgment J2 (area ambulance), dog unit DD. Zoom: G_0 none; G_1 = {T1, T2} and {A1, A2}, size 1 each; G_2 = their
sibling merge, size 1.
Checks: (1) every claim the theorems use has error 0 (district, shelter, judgment, branch; areas nested);
(2) TH1's assembly: for random rational choices per group the assembled build's logits are V C + U h_J + t_w on each
area and N's elsewhere; (3) TH2 part 1: an exact build at G_2, its restrictions are exact at G_1, and the step's new
unit is the restriction's pre-activations summed minus a constant at every point (the (r) identity)."""
import json
import os
import random
from fractions import Fraction as Fr

random.seed(0)


def relu(t):
    return t if t > 0 else Fr(0)


# points: (area, input x = (k1, k1p, k2, k2p, i3, i4, d)), weights
taxi = [(Fr(0), Fr(3)), (Fr(2), Fr(1)), (Fr(5), Fr(4))]
amb = [(Fr(1), Fr(1)), (Fr(4), Fr(0)), (Fr(2), Fr(6))]
dog = [Fr(1), Fr(3)]
P = [("taxi", [a, b, -1, -1, 2, 0, -1]) for a, b in taxi] + \
    [("amb", [-1, -1, a, b, 0, 2, -1]) for a, b in amb] + [("dog", [-1, -1, -1, -1, 0, 0, d]) for d in dog]
P = [(ar, [Fr(v) for v in x]) for ar, x in P]
pw = [Fr(1, len(P))] * len(P)
L1 = ["K1", "K1p", "K2", "K2p", "I3", "I4", "Dg"]
L2 = {  # unit: dict of weights on layer 1, bias
    "T1": ({"K1": 1, "I3": 1}, -1), "T2": ({"K1p": 1, "I3": 1}, -1), "J1": ({"I3": 1}, -1),
    "A1": ({"K2": 1, "I4": 1}, -1), "A2": ({"K2p": 1, "I4": 1}, -1), "J2": ({"I4": 1}, -1),
    "DD": ({"Dg": 1}, 0)}
alpha = {"T1": Fr(1), "T2": Fr(-2), "A1": Fr(3), "A2": Fr(1, 2), "DD": Fr(1)}   # gap weights (logit 2 is 0)


def forward(x):
    h1 = {u: relu(x[i]) for i, u in enumerate(L1)}
    z1 = {u: x[i] for i, u in enumerate(L1)}
    z2 = {u: sum(Fr(w) * h1[v] for v, w in ws.items()) + b for u, (ws, b) in L2.items()}
    h2 = {u: relu(t) for u, t in z2.items()}
    return z1, h1, z2, h2


F = [forward(x) for _, x in P]
gapN = [sum(alpha[u] * h2[u] for u in alpha) for _, _, _, h2 in F]
areas = {"taxi": {k for k, (a, _) in enumerate(P) if a == "taxi"}, "amb": {k for k, (a, _) in enumerate(P) if a == "amb"},
         "dog": {k for k, (a, _) in enumerate(P) if a == "dog"}}
areas["car"] = areas["taxi"] | areas["amb"]
allk = set(range(len(P)))

# ---- (1) claims
districts = {1: [({"K1", "K1p", "K2", "K2p", "I3", "I4"}, "car"), ({"Dg"}, "dog")],
             2: [({"T1", "T2"}, "taxi"), ({"J1"}, "taxi"), ({"A1", "A2"}, "amb"), ({"J2"}, "amb"), ({"DD"}, "dog")]}


def h(layer, u, k):
    return F[k][1][u] if layer == 1 else F[k][3][u]


def z(layer, u, k):
    return F[k][0][u] if layer == 1 else F[k][2][u]


errs = {}
for layer, ds in districts.items():
    for D, ar in ds:
        R = areas[ar]
        dist = max((sum(pw[k] * h(layer, u, k) for k in allk - R) / sum(pw[k] * h(layer, u, k) for k in allk)
                    for u in D if sum(h(layer, u, k) for k in allk) > 0), default=Fr(0))
        shel = max((len({z(layer, u, k) for k in allk - R}) > 1) * 1 for u in D)   # 0 iff constant outside
        errs[f"L{layer} {sorted(D)} dist"] = dist
        errs[f"L{layer} {sorted(D)} shelter_nonconst"] = Fr(shel)
for J, ar in (("J1", "taxi"), ("J2", "amb")):
    errs[f"judgment {J}"] = sum(pw[k] * (1 - h(2, J, k)) ** 2 for k in areas[ar])
covered = {u for D, _ in districts[1] for u in D} == set(L1) and {u for D, _ in districts[2] for u in D} == set(L2)
nested = all(areas[a2] <= areas[a1] for a2, a1 in (("taxi", "car"), ("amb", "car"), ("dog", "dog")))
# branch claims: no hidden unit reads layer 2 (L = 2), so every branch claim of a layer-2 set is 0; layer-1 sets
# are not removed by these groups.
claims_ok = all(v == 0 for v in errs.values()) and covered and nested
print("claims:", {k: str(v) for k, v in errs.items()})
print("cover", covered, "nested", nested, "CLAIMS_OK", claims_ok)

# ---- (2) TH1 assembly at G_1 with random rational choices per group (boxes may switch inside an area)
groups = {"S1": (["T1", "T2"], "taxi", "J1"), "S2": (["A1", "A2"], "amb", "J2")}


def rnd():
    return Fr(random.randint(-9, 9), random.randint(1, 4))


assembly_ok = True
for trial in range(50):
    ch = {S: dict(m=[rnd() for _ in top], beta=rnd(), V=rnd(), U=rnd(), t=rnd()) for S, (top, _, _) in groups.items()}
    box = {S: [relu(sum(c["m"][i] * z(2, u, k) for i, u in enumerate(groups[S][0])) + c["beta"]) for k in allk]
           for S, c in ch.items()}
    v = {S: ch[S]["V"] * box[S][min(allk - areas[groups[S][1]])] for S in groups}      # constant write outside
    cN = Fr(0)
    cB = cN - sum(v.values())
    for k in range(len(P)):
        gapB = cB + alpha["DD"] * h(2, "DD", k)
        for S, (top, ar, J) in groups.items():
            s_w = ch[S]["t"] - cN + v[S]
            gapB += ch[S]["V"] * box[S][k] + (ch[S]["U"] + s_w) * h(2, J, k)
        ar_k = P[k][0]
        S_here = [S for S in groups if groups[S][1] == ar_k]
        if S_here:
            S = S_here[0]
            want = ch[S]["V"] * box[S][k] + ch[S]["U"] * h(2, groups[S][2], k) + ch[S]["t"]
        else:
            want = gapN[k]
        assembly_ok &= gapB == want
print("TH1 assembly (50 random choices):", assembly_ok)

# ---- (3) TH2 part 1: exact top unit at G_2, restrictions at G_1, (r) identity
top = ["T1", "T2", "A1", "A2"]
beta = Fr(100)
m = [alpha[u] for u in top]
pre_top = [sum(m[i] * z(2, u, k) for i, u in enumerate(top)) + beta for k in allk]
assert all(p > 0 for p in pre_top)
unit_top = [relu(p) for p in pre_top]


def solve_exact(unit, groups_here):
    """Gap weight 1 on the unit; judgment weights and bias chosen so the gap equals N's on P (two logits)."""
    dk = min(areas["dog"])
    bias = gapN[dk] - unit[dk] - alpha["DD"] * h(2, "DD", dk)
    U = {}
    for J, ar in groups_here:
        k0 = min(areas[ar])
        U[J] = gapN[k0] - unit[k0] - bias          # J = 1 on its area
    gap = [unit[k] + sum(U[J] * h(2, J, k) for J in U) + alpha["DD"] * h(2, "DD", k) + bias for k in allk]
    return gap == gapN


exact_G2 = solve_exact(unit_top, [("J1", "taxi"), ("J2", "amb")])
# zeta: constant pre-activations of the other child's top outside its area
zeta = {u: z(2, u, min(areas["dog"])) for u in top}
restr = {}
for S, (tp, ar, J) in groups.items():
    other = [u for u in top if u not in tp]
    const = beta + sum(alpha[u] * zeta[u] for u in other)
    restr[S] = ([sum(alpha[u] * z(2, u, k) for u in tp) + const for k in allk])
    agree = all(relu(restr[S][k]) == unit_top[k] for k in areas[ar])
    print(f"restriction to {S} equals the top unit on its area:", agree)
# G_1 build: two restricted units, each with gap weight 1; judgment weights and bias solved as before
u1 = [relu(p) for p in restr["S1"]]
u2 = [relu(p) for p in restr["S2"]]
both = [a + b for a, b in zip(u1, u2)]
exact_G1 = solve_exact(both, [("J1", "taxi"), ("J2", "amb")])
const_r = {restr["S1"][k] + restr["S2"][k] - pre_top[k] for k in allk}
r_identity = len(const_r) == 1
print("exact at G_2:", exact_G2, " exact at G_1 (restrictions):", exact_G1, " (r) identity, constant:", r_identity,
      [str(c) for c in const_r])
# ---- (4) TH2 part 1 (a): for random LOSSY builds at G_2, the restricted build at G_1 has D2 no larger (here equal)
def d2(gap):
    return Fr(1, 4) * sum(w * (a - b) ** 2 for w, a, b in zip(pw, gap, gapN))


mono_ok = True
for trial in range(50):
    m2 = [rnd() for _ in top]
    b2 = rnd()
    V, U1, U2, t1, t2 = rnd(), rnd(), rnd(), rnd(), rnd()
    pre2 = [sum(m2[i] * z(2, u, k) for i, u in enumerate(top)) + b2 for k in allk]
    box2 = [relu(p) for p in pre2]
    v2 = V * box2[min(areas["dog"])]

    def assemble(boxvals, vconst):
        # TH1 upper bound with per-area targets t1 (taxi), t2 (amb): gap = V box + U J + t on areas, N elsewhere
        cB = -sum(vconst)
        out = []
        for k in allk:
            g = cB + alpha["DD"] * h(2, "DD", k) + sum(V * bv[k] for bv in boxvals)
            for J, U, t, vS in (("J1", U1, t1, vconst[0]), ("J2", U2, t2, vconst[-1])):
                g += (U + t + vS) * h(2, J, k)
            out.append(g)
        return out

    gap2 = assemble([box2], [v2])
    rb = {}
    for S, (tp, ar, J) in groups.items():
        other = [u for u in top if u not in tp]
        cst = b2 + sum(m2[top.index(u)] * zeta[u] for u in other)
        rb[S] = [relu(sum(m2[top.index(u)] * z(2, u, k) for u in tp) + cst) for k in allk]
    vS1, vS2 = V * rb["S1"][min(areas["dog"])], V * rb["S2"][min(areas["dog"])]
    gap1 = assemble([rb["S1"], rb["S2"]], [vS1, vS2])     # J1 shifts by S1's constant, J2 by S2's
    mono_ok &= d2(gap1) <= d2(gap2) and all(gap1[k] == gap2[k] for k in areas["car"])
print("TH2 part 1 (a): restricted G_1 build no worse than G_2 build (50 lossy choices):", mono_ok)
ALL_OK = claims_ok and assembly_ok and exact_G2 and exact_G1 and r_identity and mono_ok
print("ALL_OK", ALL_OK)
with open(os.path.join(os.path.dirname(__file__), "th_tree.json"), "w") as f:
    json.dump(dict(claims={k: str(v) for k, v in errs.items()}, assembly_ok=assembly_ok, exact_G2=exact_G2,
                   exact_G1=exact_G1, r_identity=r_identity, ALL_OK=ALL_OK), f, indent=1)
