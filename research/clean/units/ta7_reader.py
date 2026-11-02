# serves: TA7 (UNIT-BRANCH.md)
"""Check of Proposition TA7: the removal interaction of two co-firing last-layer units is, to second order,
h_u h_v w_u^T H w_v (activation x activation x reader overlap). One input, weight 1; both units on with
activation eps; logit offset chosen so that N's logits are 0 (uniform softmax, H = (1/n)(I - J/n)).
Case A, three logits, writes (1,-1,0) and (1,1,-2): orthogonal in H, ratio I/eps^2 -> 0.
Case B, three logits, writes (1,-1,0) and (1,0,-1): w_u^T H w_v = 1/3.
Case C, two logits, writes (1,0) and (1,0): s1 s2 = 1/4.
Last line: in case A, TU7's free write still helps (gamma_v != 0), so the reader factor vanishes again."""
import json
import math
import os


def kl0(d):                     # KL(uniform || softmax(d))
    n = len(d); m = max(d); lse = m + math.log(sum(math.exp(t - m) for t in d))
    return sum((1 / n) * (math.log(1 / n) - (t - lse)) for t in d)


def I_rem(wu, wv, eps):
    a = [-eps * t for t in wu]; b = [-eps * t for t in wv]
    return kl0([x + y for x, y in zip(a, b)]) - kl0(a) - kl0(b)


def hform(wu, wv):
    n = len(wu)
    return sum(x * y for x, y in zip(wu, wv)) / n - sum(wu) * sum(wv) / n ** 2


cases = dict(A=((1, -1, 0), (1, 1, -2)), B=((1, -1, 0), (1, 0, -1)), C=((1, 0), (1, 0)))
out = {}
for k, (wu, wv) in cases.items():
    out[k] = dict(predicted_second_order=hform(wu, wv),
                  ratio={str(e): I_rem(wu, wv, e) / e ** 2 for e in (0.1, 0.01, 0.001)})
# TU7 in case A: S = {u} removed (eps = 1); helper relu(z_v) with a free write w; gamma_v = h_v (softmax(-w_u) - uniform)
wu = cases["A"][0]
z = [math.exp(-t) for t in wu]; s = [t / sum(z) for t in z]
gamma = [si - 1 / 3 for si in s]
out["A_free_write_gamma_v"] = gamma
ok = (abs(out["A"]["ratio"]["0.001"]) < 1e-3 and abs(out["B"]["ratio"]["0.001"] - 1 / 3) < 1e-3
      and abs(out["C"]["ratio"]["0.001"] - 0.25) < 1e-3 and max(abs(g) for g in gamma) > 0.1)
out["prediction"] = "A -> 0, B -> 1/3, C -> 1/4; gamma_v in case A non-zero"
out["held"] = ok
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ta7_reader.json"), "w"), indent=1)
