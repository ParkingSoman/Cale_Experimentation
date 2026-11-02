# serves: TM1
# v4 adds TM1 part 4: a unit f in no district (flat on each region, or varying on region B as a control).
"""Exact check of Theorem TM1 (COARSE-THEORY.md B.M), version 3: box by box.
TM1's proof holds for every FIXED pair of boxes: the best free logits of the joint build cost exactly the sum of the
two groups' region terms l_S(C_S) + l_T(C_T). With the boxes fixed, the logits are the only free part and the fit is
convex (multinomial logistic with soft targets), solved here by Newton's method to ~1e-12.
Input x = (1A, uA, vA, 1B, uB, vB). Region A: NPR points with 1A = 1; region B: NPR points with 1B = 1; one point in
no region (all zeros). One hidden layer (= the last hidden layer), 3 logits.
Districts: S = {a1, a2, a3} (region A), {jA} judgment of A, {kA} (A); T = {b1, b2, b3} (region B), {jB} judgment of B.
Versions 1 and 2 (random-restart L-BFGS over everything) were too weak: see UNIT-LOG.md."""
import json
import os

import numpy as np

rng = np.random.default_rng(0)
NPR = 8
A_pts = rng.uniform(0, 1, size=(NPR, 2))
B_pts = rng.uniform(0, 1, size=(NPR, 2))
X = np.zeros((2 * NPR + 1, 6))
X[:NPR, 0] = 1
X[:NPR, 1:3] = A_pts
X[NPR:2 * NPR, 3] = 1
X[NPR:2 * NPR, 4:6] = B_pts
p = np.ones(len(X)) / len(X)
inA = np.r_[np.ones(NPR), np.zeros(NPR + 1)].astype(bool)
inB = np.r_[np.zeros(NPR), np.ones(NPR), 0].astype(bool)
NAMES = ["a1", "a2", "a3", "jA", "kA", "b1", "b2", "b3", "jB"]
S = ["a1", "a2", "a3"]
T = ["b1", "b2", "b3"]
Wout = rng.normal(size=(3, 9)) * 1.5
cN = rng.normal(size=3)
w_f = rng.normal(size=3) * 1.5   # N's write of the extra unit f (v4)


def relu(t):
    return np.maximum(t, 0.0)


def pre(shelter_broken=False, flat=None):
    oA, uA, vA, oB, uB, vB = X.T
    z = {"a1": 2 * uA - vA - 0.3 * oA, "a2": -uA + 2 * vA - 0.4 * oA, "a3": uA + vA - 0.9 * oA, "jA": oA,
         "kA": uA - 0.1 * oA, "b1": 2 * uB - vB - 0.5 * oB, "b2": -uB + 3 * vB - 0.6 * oB,
         "b3": 3 * uB + vB - 2.2 * oB, "jB": oB}
    if shelter_broken:
        z["a1"] = z["a1"] + 0.8 * uB - 2.0 * oB    # still off on B (z < 0 there) but varies with uB there
    if flat == "flat":
        z["f"] = 0.7 * oA + 0.2 * oB + 0.1         # 0.8 on A, 0.3 on B, 0.1 on the point in no region
    elif flat == "vary":
        z["f"] = 0.7 * oA + 0.5 * uB + 0.1         # varies on region B
    return z


def softmax(y):
    y = y - y.max(1, keepdims=True)
    e = np.exp(y)
    return e / e.sum(1, keepdims=True)


def fit(q, F, w):
    """min over Theta of sum_x w_x KL(q_x || softmax(F_x Theta)), class 3 as reference (logit 0). Newton."""
    n, d = F.shape
    th = np.zeros((d, 2))
    for _ in range(100):
        y = np.c_[F @ th, np.zeros(n)]
        s = softmax(y)
        r = (s - q)[:, :2] * w[:, None]
        g = (F.T @ r).ravel()
        H = np.zeros((2 * d, 2 * d))
        for i in range(n):
            si = s[i, :2]
            M = (np.diag(si) - np.outer(si, si)) * w[i]
            H += np.kron(np.outer(F[i], F[i]), M)
        H += 1e-13 * np.eye(2 * d)
        step = np.linalg.solve(H, g).reshape(d, 2)
        th -= step
        if np.abs(step).max() < 1e-13:
            break
    y = np.c_[F @ th, np.zeros(n)]
    lq = np.log(np.clip(q, 1e-300, None))
    ls = np.log(np.clip(softmax(y), 1e-300, None))
    return float(np.sum(w * np.sum(q * (lq - ls), 1)))


def box(z, group, mb):
    Z = np.stack([z[n] for n in group], 1)
    return relu(Z @ mb[:-1] + mb[-1])


def check(shelter_broken=False, judgment_kept=True, n_pairs=200, flat=None):
    z = pre(shelter_broken, flat)
    H = np.stack([relu(z[n]) for n in NAMES], 1)
    yN = H @ Wout.T + cN
    if flat:
        yN = yN + np.outer(relu(z["f"]), w_f)
    q = softmax(yN)
    kept = ["jA", "kA", "jB"] if judgment_kept else ["kA"]
    extra = ["f"] if flat else []
    KS = [n for n in kept if n in ("jA", "kA")]
    KT = [n for n in kept if n == "jB"]
    one = np.ones((len(X), 1))
    dev_joint, dev_S, dev_T, rows = 0.0, 0.0, 0.0, []
    for _ in range(n_pairs):
        mS, mT = rng.normal(size=4) * 2, rng.normal(size=4) * 2
        CS, CT = box(z, S, mS), box(z, T, mT)
        hk = lambda names: np.stack([relu(z[n]) for n in names], 1) if names else np.zeros((len(X), 0))
        # joint build: both boxes, kept units, free logits (weights + bias)
        J = fit(q, np.c_[CS, CT, hk(kept + extra), one], p)
        # single builds: the other group kept with free weights
        sS = fit(q, np.c_[CS, hk(kept + T + extra), one], p)
        sT = fit(q, np.c_[CT, hk(kept + S + extra), one], p)
        # region terms of TM1: box, the kept units of the group's region, one free offset on the region
        lS = fit(q, np.c_[CS, hk(KS), one], p * inA)
        lT = fit(q, np.c_[CT, hk(KT), one], p * inB)
        dev_joint = max(dev_joint, abs(J - lS - lT))
        dev_S = max(dev_S, abs(sS - lS))
        dev_T = max(dev_T, abs(sT - lT))
        rows.append((J, sS, sT, lS, lT))
    rows = np.array(rows)
    return dict(max_dev_joint=dev_joint, max_dev_single_S=dev_S, max_dev_single_T=dev_T,
                mean_joint=float(rows[:, 0].mean()), mean_lS=float(rows[:, 3].mean()), mean_lT=float(rows[:, 4].mean()),
                min_I_fixed=float((rows[:, 0] - rows[:, 1] - rows[:, 2]).min()),
                max_I_fixed=float((rows[:, 0] - rows[:, 1] - rows[:, 2]).max()))


res = dict(main=check(), shelter_broken=check(shelter_broken=True), no_judgment=check(judgment_kept=False),
           flat_unit=check(flat="flat"), varying_unit=check(flat="vary"))
print(json.dumps(res, indent=1))
m = res["main"]
ok = max(m["max_dev_joint"], m["max_dev_single_S"], m["max_dev_single_T"]) <= 1e-9
print("PREDICTION main (J = l_S + l_T, single_S = l_S, single_T = l_T, to 1e-9, 200 box pairs):",
      "HELD" if ok else "FAILED")
sb = res["shelter_broken"]
print("PREDICTION shelter broken (max deviation > 1e-4):",
      "HELD" if max(sb["max_dev_joint"], sb["max_dev_single_S"], sb["max_dev_single_T"]) > 1e-4 else "FAILED")
nj = res["no_judgment"]
print("PREDICTION no judgment (max deviation > 1e-4):",
      "HELD" if max(nj["max_dev_joint"], nj["max_dev_single_S"], nj["max_dev_single_T"]) > 1e-4 else "FAILED")
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tm1_split.json"), "w"), indent=1)
fl, va = res["flat_unit"], res["varying_unit"]
print("PREDICTION v4 flat unit (split holds, < 1e-9):",
      "HELD" if max(fl["max_dev_joint"], fl["max_dev_single_S"], fl["max_dev_single_T"]) <= 1e-9 else "FAILED")
print("PREDICTION v4 unit varying on B (deviation > 1e-4):",
      "HELD" if max(va["max_dev_joint"], va["max_dev_single_S"], va["max_dev_single_T"]) > 1e-4 else "FAILED")
