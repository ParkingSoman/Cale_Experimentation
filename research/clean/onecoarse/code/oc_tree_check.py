# scale: neurons=s3:69 children + 3 planted units; net=A,B; photos=4,000 test (same as oc_tree.py); where=mac
# rung: Q=CALC 3.7 repair after review: neutral null, planted test, C5 fair null; EST=1 min Mac; PERMIT=Shonu approved plan 2026-11-25
"""CALC 3.7 repair (post-review, labelled as such). The registered run (oc_tree_tests.py) used a null that reads s3:69
NEGATIVELY (R1 score below the median, a signed score). This script:
 1. NEUTRAL NULL: the 50 non-child A neurons with the smallest |R1 score| (seed-free). Reruns C1, C3, C4 and the
    within-parent C4 with it, and prints base rates.
 2. PLANTED TEST: 3 synthetic children = a real child's read with its read on sibling j's 3 condition channels set to
    minus its own largest channel read (a strong down vote), bn of the real child. Pairs (305->97), (108->163), (275->297).
    C2, C3 and C4 (within parent) must flag each planted unit against sibling j; if not, the tool cannot see turning away.
 3. C2 also with the MIN vote over j's 3 channels (one down vote is enough), beside the registered mean.
 4. C5 against B's next 10 readers of s3:69 (n101 was picked as B's top reader of s3:69).
Output: results/oc_tree_check.json
"""
import itertools
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(CLEAN, "curves", "map_agents", "resnet18"))
sys.path.insert(0, HERE)
import net  # noqa: E402
from oc_tree_tests import KIDS, Q, poles, read_vec  # noqa: E402

torch.set_grad_enabled(False)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
RES = os.path.join(HERE, "..", "results")
WD = os.path.join(CLEAN, "neuron", "results", "coarse", "r18")
PLANT = [(305, 97), (108, 163), (275, 297)]


def planted_pre(ids, W, bn, cond):
    ph = net.Photos("read"); A = net.load().to(DEV)
    g, b, m, v = bn
    Ws, out = [], []
    for i, j in PLANT:
        w = W[i].copy()
        rv = w.sum((1, 2)) * g[i] / np.sqrt(v[i] + 1e-5)
        target = -np.abs(rv).max()
        for c in cond[j]:
            w[c] += (target - rv[c]) / (9 * g[i] / np.sqrt(v[i] + 1e-5))
        Ws.append(w)
    Wt = torch.tensor(np.stack(Ws), dtype=torch.float32, device=DEV)
    idx = [i for i, _ in PLANT]
    sc = torch.tensor(g[idx] / np.sqrt(v[idx] + 1e-5), dtype=torch.float32, device=DEV)
    sh = torch.tensor(b[idx] - m[idx] * g[idx] / np.sqrt(v[idx] + 1e-5), dtype=torch.float32, device=DEV)
    for a in range(0, len(ids), 100):
        x = net.upto(A, ph.tensor(ids[a:a + 100]).to(DEV), 4)
        y = F.conv2d(x, Wt, stride=2, padding=1) * sc[None, :, None, None] + sh[None, :, None, None]
        out.append(y.flatten(2).cpu())
    return torch.cat(out).numpy(), np.stack(Ws)


def main():
    D = np.load(os.path.join(RES, "oc_tree.npz"))
    A, B, PW, ids = D["A_pre"].astype(np.float32), D["B_pre"].astype(np.float32), D["pwin"], D["ids"]
    on = PW >= np.percentile(PW, 80); base_on = float(on.mean())
    cw = np.load(os.path.join(RES, "oc_children_w.npz"))
    cand = np.array([j for j in range(512) if j not in KIDS])
    NULL = cand[np.argsort(np.abs(cw["score"][cand]))[:50]]
    wA = np.load(os.path.join(WD, "w_A.npz")); RA = read_vec(wA["W1_0"], wA["bn1_0"])
    ch = np.array([c for c in range(256) if c != 69])
    cond = {j: ch[np.argsort(-RA[j, ch])[:3]] for j in KIDS}
    Pp, Wp = planted_pre(ids, wA["W1_0"], wA["bn1_0"], cond)
    # check: an unplanted copy would equal the real child (planted channels are the only change)
    R = {"null_neurons": NULL.tolist(), "null_abs_score_max": float(np.abs(cw["score"][NULL]).max()), "base_parent_on": base_on}
    units = {j: A[:, j] for j in list(KIDS) + list(NULL)}
    for n, (i, j) in enumerate(PLANT):
        units[f"P{i}-{j}"] = Pp[:, n]
    P = {u: poles(x) for u, x in units.items()}
    share_on = lambda c: float(on[c[0], c[1]].mean())
    # C1, C4 with the neutral null
    for cid, k in (("C1", 0), ("C4", 1)):
        nul = np.array([share_on(P[j][k]) for j in NULL]); p95 = float(np.percentile(nul, 95))
        kid = {str(j): share_on(P[j][k]) for j in KIDS}
        R[cid] = dict(kids=kid, null_median=float(np.median(nul)), null_p95=p95, base=base_on,
                      n_pass=int(sum(v > p95 for v in kid.values())),
                      planted={f"P{i}-{j}": share_on(P[f"P{i}-{j}"][k]) for i, j in PLANT})
    # C3 and within-parent C4 with the neutral null
    pc, pr = np.where(on)
    val = lambda u: units[u][pc, pr]
    top10 = {u: val(u) >= np.percentile(val(u), 90) for u in KIDS}
    def d_of(xi, xj):
        J = xj >= np.percentile(xj, 90); return float(np.median(xi[J]) - np.median(xi))
    c3, c3p = [], {}
    nul_cache = {}
    for i in list(KIDS) + [f"P{a}-{b}" for a, b in PLANT]:
        xi = val(i)
        nul = np.array([d_of(xi, val(r)) for r in NULL]); p10 = float(np.percentile(nul, 10))
        for j in KIDS:
            if (j == i):
                continue
            if isinstance(i, str) and j != int(i.split("-")[1]):
                continue
            d = d_of(xi, val(j)); row = dict(i=str(i), j=j, d=d, null_p10=p10, null_median=float(np.median(nul)), away=bool(d < p10))
            (c3p.__setitem__(i, row) if isinstance(i, str) else c3.append(row))
    R["C3"] = dict(n_away=int(sum(r["away"] for r in c3)), n_pairs=len(c3), median_d=float(np.median([r["d"] for r in c3])), planted=c3p)
    def lift(i, sibs):
        ng = np.argsort(val(i))[:Q]; anys = np.zeros(len(pc), bool)
        for j in sibs: anys |= top10[j]
        return float(anys[ng].mean() / anys.mean())
    rng = np.random.default_rng(0)
    nl = np.array([lift(r, list(rng.choice(KIDS, 10, replace=False))) for r in NULL])
    kl = {str(i): lift(i, [j for j in KIDS if j != i]) for i in KIDS}
    def lift1(i, j):  # planted: lift on the ONE targeted sibling's top cells
        ng = np.argsort(val(i))[:Q]; return float(top10[j][ng].mean() / top10[j].mean())
    nl1 = np.array([lift1(r, int(rng.choice(KIDS))) for r in NULL])
    R["C4in"] = dict(kids=kl, null_median=float(np.median(nl)), null_p95=float(np.percentile(nl, 95)),
                     n_above=int(sum(v > np.percentile(nl, 95) for v in kl.values())),
                     planted={f"P{i}-{j}": lift1(f"P{i}-{j}", j) for i, j in PLANT},
                     planted_real_child={f"{i}-{j}": lift1(i, j) for i, j in PLANT},
                     single_null_median=float(np.median(nl1)), single_null_p95=float(np.percentile(nl1, 95)))
    # C2: registered mean and the min version; planted reads
    RP = (Wp.sum((2, 3)) * np.array([wA["bn1_0"][0][i] / np.sqrt(wA["bn1_0"][3][i] + 1e-5) for i, _ in PLANT])[:, None])
    def c2(vec, j, agg):
        vote = float(agg(vec[cond[j]])); trip = np.array([agg(vec[rng.choice(ch, 3, replace=False)]) for _ in range(2000)])
        return vote, float((trip < vote).mean())
    for name, agg in (("mean", np.mean), ("min", np.min)):
        rows = [dict(i=i, j=j, **dict(zip(("vote", "pct"), c2(RA[i], j, agg)))) for i, j in itertools.permutations(KIDS, 2)]
        R[f"C2_{name}"] = dict(n_away=int(sum(r["pct"] < .25 for r in rows)), n_pairs=len(rows),
                              planted={f"P{i}-{j}": c2(RP[n], j, agg)[1] for n, (i, j) in enumerate(PLANT)},
                              cited={f"{i}-{j}": c2(RA[i], j, agg)[1] for i, j in ((23, 297), (385, 163))})
    # C5 fair null: B's next 10 readers of s3:69
    wB = np.load(os.path.join(WD, "w_B128aug_s1.npz")); RB = read_vec(wB["W1_0"], wB["bn1_0"])
    rd = np.argsort(-RB[:, 69])
    U = set()
    for j in KIDS:
        U |= set(zip(*P[j][0]))
    U = np.array(sorted(U)); thr = np.percentile(B, 90, axis=(0, 2))
    cov = {int(b): float((B[U[:, 0], b, U[:, 1]] >= thr[b]).mean()) for b in rd[:11]}
    R["C5"] = dict(top_readers_of_69=[int(b) for b in rd[:11]], coverage=cov, n101_is_top=bool(rd[0] == 101))
    json.dump(R, open(os.path.join(RES, "oc_tree_check.json"), "w"), indent=1)
    print("base parent-on", round(base_on, 3), "| neutral null |score| max", round(R["null_abs_score_max"], 3))
    for k in ("C1", "C4"):
        r = R[k]; print(k, "kids above null p95:", r["n_pass"], "of 11 | null median", round(r["null_median"], 3), "p95", round(r["null_p95"], 3),
                        "| kids", [round(v, 2) for v in r["kids"].values()], "| planted", {a: round(b, 2) for a, b in r["planted"].items()})
    print("C3 pairs away:", R["C3"]["n_away"], "of", R["C3"]["n_pairs"], "median d", round(R["C3"]["median_d"], 3),
          "| planted", {a: (round(b["d"], 3), b["away"]) for a, b in R["C3"]["planted"].items()})
    r = R["C4in"]; print("C4 within parent: kids above null p95", r["n_above"], "| null median", round(r["null_median"], 2), "p95", round(r["null_p95"], 2),
                         "| kids", [round(v, 2) for v in r["kids"].values()])
    print("   planted lift on target sibling", {a: round(b, 2) for a, b in r["planted"].items()}, "real child", {a: round(b, 2) for a, b in r["planted_real_child"].items()},
          "| single-sibling null median", round(r["single_null_median"], 2), "p95", round(r["single_null_p95"], 2))
    for name in ("mean", "min"):
        r = R[f"C2_{name}"]; print(f"C2 {name}: pairs away", r["n_away"], "of 110 | planted pct", {a: round(b, 3) for a, b in r["planted"].items()},
                                   "| cited pct", {a: round(b, 3) for a, b in r["cited"].items()})
    print("C5: B top readers of s3:69", R["C5"]["top_readers_of_69"], "coverage", {a: round(b, 2) for a, b in R["C5"]["coverage"].items()})


if __name__ == "__main__":
    main()
