"""HOW round 6: do merges help as a STARTING POINT for a short short-tail fit (blue sky #6)?

Same budget for every start: `--minutes` of short-tail KL fitting on prose (r2round5.fit_tail), K = 48, layer L.
Starts:
  CUT          48 old neurons with the biggest output, joint LS output refit
  KIDS-H0      dominant child of each verified family (what from the trained layer), joint LS refit
  KIDS-AVG     average of each family's children, joint LS refit
  KIDS-H6      M1 local fit per family, joint LS refit
  CLUSTER-AVG  NO trained layer: k-means of old neurons by where they fire (sign-aligned activation profiles on
               calibration text), average each cluster (tree_lib.merge_groups), joint LS refit
  RANDOM       random weights
  python3 research/clean/r2round6.py --layer 28 --minutes 1
"""
import argparse
import json
import os
import time

import numpy as np

from r2lib import Setup, ROOT
from r2round3 import joint_out
from r2round5 import cache_h, fit_tail
from r2round2 import ce_blocks
from r1 import SmallMLP, rand_init
from m1 import run_rules
from tree_lib import token_blocks, canonical_sign, neuron_features, kmeans, merge_groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--minutes", type=float, default=1)
    ap.add_argument("--starts", default="CUT,KIDS-H0,KIDS-AVG,KIDS-H6,CLUSTER-AVG,RANDOM")
    args = ap.parse_args()
    S = Setup(args.layer, args.K, "global")
    t0 = time.time()
    K = args.K
    Yc = S.Hc @ S.D.T
    H = cache_h(S, token_blocks(S.tok, "train", 64))
    held = token_blocks(S.tok, "test", 100)
    orig = S.model.model.layers[S.L].mlp
    base = float(ce_blocks(S.model, held).mean())

    def ce_of(P):
        S.model.model.layers[S.L].mlp = SmallMLP(P)
        r = float(ce_blocks(S.model, held).mean())
        S.model.model.layers[S.L].mlp = orig
        return r

    fams = {f["unit"]: f for f in S.families}
    from tree_lib import hidden
    aT = hidden(S.Xc, S.P[0], S.P[1])
    Hz = (S.Hc - S.Hc.mean(0)) / (S.Hc.std(0) + 1e-9)

    def best_old(t):
        a = (aT[:, t] - aT[:, t].mean()) / (aT[:, t].std() + 1e-9)
        c = Hz.T @ a / len(a)
        j = int(np.argmax(np.abs(c)))
        return S.G[j], float(np.sign(c[j])) * S.U[j]

    def kids(rule):
        G, U = np.zeros((K, 576)), np.zeros((K, 576))
        for t in range(K):
            if t in fams:
                f = fams[t]
                P = run_rules(S.Xc, S.Xh, f["Gc"], f["Uc"], f["Dc"], 1, rules=[rule])[rule][0]
                G[t], U[t] = P[0][0], P[1][0]
            else:
                G[t], U[t] = best_old(t)
        return G, U

    starts = {}
    names = args.starts.split(",")
    if "CUT" in names:
        imp = np.abs(S.Hc).mean(0) * np.linalg.norm(S.D, axis=0)
        keep = np.sort(np.argsort(-imp)[:K])
        starts["CUT"] = (S.G[keep], S.U[keep])
    for rule, name in (("H0", "KIDS-H0"), ("H1", "KIDS-AVG"), ("H6", "KIDS-H6")):
        if name in names:
            starts[name] = kids(rule)
    if "CLUSTER-AVG" in names:
        sign = canonical_sign(S.Xc, S.G, S.U)
        feat = neuron_features("function", S.Xc, S.G, S.U, S.D, sign)
        lab = kmeans(feat, K, seed=0)
        G2, U2, _ = merge_groups(S.G, S.U, S.D, sign, lab)
        starts["CLUSTER-AVG"] = (G2.astype(np.float64), U2.astype(np.float64))
    res = {"layer": S.L, "K": K, "minutes": args.minutes, "base": base, "rows": {}}
    for name, (G, U) in starts.items():
        D = joint_out(S.Xc, G, U, Yc)
        P0 = (G, U, D)
        if name == "RANDOM":
            P0 = rand_init(S.G, S.U, S.D, K, seed=0)
        ce0 = ce_of(P0)
        P, steps = fit_tail(S, H, P0, args.minutes)
        ce1 = ce_of(P)
        res["rows"][name] = {"ce_start": ce0, "ce_after": ce1, "steps": steps}
        print(f"{name:12s} start +{ce0 - base:.4f} -> after {args.minutes} min +{ce1 - base:.4f} ({steps} steps) "
              f"[{time.time() - t0:.0f}s]", flush=True)
        json.dump(res, open(os.path.join(ROOT, "results", f"clean-r2round6-L{S.L}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
