"""HOW round 8 (hybrid): children set WHERE a parent fires, the objective sets WHAT it writes?

Round 3 split the gap to the trained layer into gates/values (0.034) and outputs (0.033). Here the gates (and
optionally the values) come from a start and stay FROZEN. The short-tail fit (r2round5) trains only the rest.
Starts: CUT, KIDS-H6 (children-built parents, what from the trained layer), RANDOM.
Modes: full (all parts trained), UD (gate frozen), D (gate and value frozen).

  python3 research/clean/r2round8.py --layer 28 --minutes 1
"""
import argparse
import json
import os
import time

import numpy as np

from r2lib import Setup, ROOT, hidden
from r2round3 import joint_out
from r2round5 import cache_h, fit_tail
from r2round2 import ce_blocks
from r1 import SmallMLP, rand_init
from m1 import run_rules
from tree_lib import token_blocks

MODES = {"full": (), "UD": ("gate_proj",), "D": ("gate_proj", "up_proj")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--minutes", type=float, default=1)
    ap.add_argument("--runs", default="CUT:D,CUT:UD,KIDS-H6:D,KIDS-H6:UD,RANDOM:full")
    args = ap.parse_args()
    S = Setup(args.layer, args.K, "global")
    t0 = time.time()
    K = args.K
    H = cache_h(S, token_blocks(S.tok, "train", 64))
    held = token_blocks(S.tok, "test", 100)
    orig = S.model.model.layers[S.L].mlp
    base = float(ce_blocks(S.model, held).mean())

    def ce_of(P):
        S.model.model.layers[S.L].mlp = SmallMLP(P)
        r = float(ce_blocks(S.model, held).mean())
        S.model.model.layers[S.L].mlp = orig
        return r

    Yc = S.Hc @ S.D.T
    imp = np.abs(S.Hc).mean(0) * np.linalg.norm(S.D, axis=0)
    keep = np.sort(np.argsort(-imp)[:K])
    starts = {"CUT": (S.G[keep], S.U[keep])}
    fams = {f["unit"]: f for f in S.families}
    aT = hidden(S.Xc, S.P[0], S.P[1])
    Hz = (S.Hc - S.Hc.mean(0)) / (S.Hc.std(0) + 1e-9)
    G, U = np.zeros((K, 576)), np.zeros((K, 576))
    for t in range(K):
        if t in fams:
            f = fams[t]
            P = run_rules(S.Xc, S.Xh, f["Gc"], f["Uc"], f["Dc"], 1, rules=["H6"])["H6"][0]
            G[t], U[t] = P[0][0], P[1][0]
        else:
            a = (aT[:, t] - aT[:, t].mean()) / (aT[:, t].std() + 1e-9)
            c = Hz.T @ a / len(a)
            j = int(np.argmax(np.abs(c)))
            G[t], U[t] = S.G[j], float(np.sign(c[j])) * S.U[j]
    starts["KIDS-H6"] = (G, U)
    res = {"layer": S.L, "K": K, "minutes": args.minutes, "base": base, "rows": {}}
    for run in args.runs.split(","):
        sname, mode = run.split(":")
        if sname == "RANDOM":
            P0 = rand_init(S.G, S.U, S.D, K, seed=0)
        else:
            Gs, Us = starts[sname]
            P0 = (Gs, Us, joint_out(S.Xc, Gs, Us, Yc))
        ce0 = ce_of(P0)
        P, steps = fit_tail(S, H, P0, args.minutes, frozen=MODES[mode])
        ce1 = ce_of(P)
        res["rows"][run] = {"ce_start": ce0, "ce_after": ce1, "steps": steps}
        print(f"{sname:8s} {mode:4s} (frozen {MODES[mode]}) start +{ce0 - base:.4f} -> +{ce1 - base:.4f} "
              f"({steps} steps) [{time.time() - t0:.0f}s]", flush=True)
        json.dump(res, open(os.path.join(ROOT, "results", f"clean-r2round8-L{S.L}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
