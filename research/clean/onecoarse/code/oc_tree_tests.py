"""CALC 3.7 tests C1-C5, exactly as pre-registered in distill/CALC.md 3.7. numpy only, from results/oc_tree.npz,
results/oc_children_w.npz, and the saved weights w_A.npz, w_B128aug_s1.npz. Output: results/oc_tree_tests.json"""
import itertools
import json
import os

import numpy as np
from scipy.stats import binom

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
WD = os.path.join(HERE, "..", "..", "neuron", "results", "coarse", "r18")
KIDS = [305, 97, 23, 163, 385, 297, 464, 237, 108, 510, 275]
Q = 200


def read_vec(w, bn):
    g, _, _, v = bn
    return w.sum((2, 3)) * (g / np.sqrt(v + 1e-5))[:, None]  # neurons x 256 channels


def poles(X):  # X: (N, 49) -> top cells and negative cells as (photo, cell)
    mx, mn = X.max(1), X.min(1)
    tp = np.argsort(-mx)[:Q]; ng = np.argsort(mn)[:Q]
    return (tp, X[tp].argmax(1)), (ng, X[ng].argmin(1))


def main():
    D = np.load(os.path.join(RES, "oc_tree.npz"))
    A, B, PW = D["A_pre"].astype(np.float32), D["B_pre"].astype(np.float32), D["pwin"]
    N = len(PW)
    on = PW >= np.percentile(PW, 80)
    cw = np.load(os.path.join(RES, "oc_children_w.npz"))
    rng = np.random.default_rng(3)
    low = np.array([j for j in range(512) if cw["score"][j] < np.median(cw["score"]) and j not in KIDS])
    NULL = rng.choice(low, 50, replace=False)
    R = {"null_neurons": NULL.tolist()}

    def share_on(cells):
        return float(on[cells[0], cells[1]].mean())

    P = {j: poles(A[:, j]) for j in list(KIDS) + list(NULL)}
    # C1, C4
    for cid, k in (("C1", 0), ("C4", 1)):
        nul = np.array([share_on(P[j][k]) for j in NULL]); p95 = float(np.percentile(nul, 95))
        kid = {j: share_on(P[j][k]) for j in KIDS}
        n = sum(v > p95 for v in kid.values())
        R[cid] = dict(kids=kid, null_p95=p95, null_median=float(np.median(nul)), n_pass=int(n), passes=bool(n >= 8))
    # C2
    wA = np.load(os.path.join(WD, "w_A.npz"))
    RA = read_vec(wA["W1_0"], wA["bn1_0"])
    ch = np.array([c for c in range(256) if c != 69])
    cond = {j: ch[np.argsort(-RA[j, ch])[:3]] for j in KIDS}
    pairs = []
    for i, j in itertools.permutations(KIDS, 2):
        vote = float(RA[i, cond[j]].mean())
        trip = np.array([RA[i, rng.choice(ch, 3, replace=False)].mean() for _ in range(2000)])
        pairs.append(dict(i=i, j=j, vote=vote, p25=float(np.percentile(trip, 25)), pct=float((trip < vote).mean()),
                          away=bool(vote < np.percentile(trip, 25))))
    na = sum(p["away"] for p in pairs)
    R["C2"] = dict(conditions={j: cond[j].tolist() for j in KIDS}, n_away=int(na), n_pairs=len(pairs),
                   n_negative_vote=int(sum(p["vote"] < 0 for p in pairs)), p_binom=float(binom.sf(na - 1, len(pairs), .25)),
                   passes=bool(na >= 36), pairs=pairs)
    # C3
    pc, pr = np.where(on)
    def d_of(i, j):
        xj = A[pc, j, pr] if j is not None else None
        J = xj >= np.percentile(xj, 90)
        xi = A[pc, i, pr]
        return float(np.median(xi[J]) - np.median(xi))
    c3 = []
    for i in KIDS:
        nul = np.array([d_of(i, r) for r in NULL]); p10 = float(np.percentile(nul, 10))
        for j in KIDS:
            if j == i:
                continue
            d = d_of(i, j)
            c3.append(dict(i=i, j=j, d=d, null_p10=p10, null_median=float(np.median(nul)), away=bool(d < p10)))
    n3 = sum(p["away"] for p in c3)
    R["C3"] = dict(n_away=int(n3), n_pairs=len(c3), passes=bool(n3 >= 55), pairs=c3)
    # C5
    U = set()
    for j in KIDS:
        U |= set(zip(*P[j][0]))
    U = np.array(sorted(U)); thr = np.percentile(B, 90, axis=(0, 2))
    cov = lambda b: float((B[U[:, 0], b, U[:, 1]] >= thr[b]).mean())
    others = np.array([cov(b) for b in range(128) if b != 101])
    wB = np.load(os.path.join(WD, "w_B128aug_s1.npz"))
    RB = read_vec(wB["W1_0"], wB["bn1_0"])
    rank69 = int((RB[101] > RB[101, 69]).sum()) + 1
    votes = {j: float(RB[101, cond[j]].mean()) for j in KIDS}
    nneg = sum(v < 0 for v in votes.values())
    c101 = cov(101)
    R["C5"] = dict(union_cells=len(U), coverage=c101, others_p95=float(np.percentile(others, 95)),
                   others_median=float(np.median(others)), rank_ch69=rank69, votes_on_child_conditions=votes,
                   n_negative=int(nneg),
                   passes=bool(c101 >= .5 and c101 > np.percentile(others, 95) and rank69 <= 10 and nneg <= 5))
    json.dump(R, open(os.path.join(RES, "oc_tree_tests.json"), "w"), indent=1)
    for k in ("C1", "C4"):
        print(k, "pass" if R[k]["passes"] else "FAIL", R[k]["n_pass"], "of 11; null p95", round(R[k]["null_p95"], 3),
              {j: round(v, 3) for j, v in R[k]["kids"].items()})
    print("C2", "pass" if R["C2"]["passes"] else "FAIL", R["C2"]["n_away"], "of 110 (bar 36); negative votes", R["C2"]["n_negative_vote"])
    print("C3", "pass" if R["C3"]["passes"] else "FAIL", R["C3"]["n_away"], "of 110 (bar 55)")
    c = R["C5"]
    print("C5", "pass" if c["passes"] else "FAIL", "coverage", round(c["coverage"], 3), "others p95", round(c["others_p95"], 3),
          "rank ch69", c["rank_ch69"], "negative votes", c["n_negative"])


if __name__ == "__main__":
    main()
