"""HOW round 3: build the WHOLE small layer from children, then refit all outputs jointly (E6, E7).

E6: one-at-a-time swaps inside a co-trained layer are biased against any rule (neighbours already share the job).
E7: so build all K parents with one rule, then refit every output column together by least squares to the original
    layer output (plain, or under the loss-aware Fisher metric). Score = held-out loss with that layer in place.

"What" (which children) comes from the trained layer on purpose: verified families where they exist, else the old
neuron that best matches the trained unit. That isolates "how" at the layer level.

Layers compared (all K = 48 neurons, layer L only replaced):
  CUT            keep the 48 old neurons with the biggest output, joint output refit        (no families)
  KIDS-H0        each parent = its dominant child, joint refit                              (a smarter cut)
  KIDS-AVG       each parent = average of its children (M1 H1), joint refit
  KIDS-H6        each parent = M1 local gradient fit to the children's summed output, joint refit
  KIDS-PROBE     union-probe gate (round 1), value fitted to the children, joint refit
  TRAINED-GU     the trained units' gates and values, joint refit                          (reference)
  TRAINED        the trained (global) layer as is                                           (reference)
A joint refit with a free output matrix gives the same answer under any output metric, so there is one form.

  python3 research/clean/r2round3.py --layer 28
"""
import argparse
import json
import os
import time

import numpy as np
import mlx.core as mx

from r2lib import Setup, union_labels, fit_value_out, ROOT, silu, hidden
from r2round1 import probe_ridge
from r2round2 import fisher, ce_blocks
from r1 import SmallMLP
from m1 import run_rules


def joint_out(X, G, U, Y, R=None, ridge=1e-4):
    """All output columns at once: D minimizing ||(Y - H D^T) R^T||, H = hidden(X, G, U). Returns D (576 x K)."""
    H = hidden(X, G, U)
    A = H.T @ H
    A += np.eye(A.shape[0]) * ridge * np.trace(A) / A.shape[0]
    D = np.linalg.solve(A, H.T @ Y).T                     # least squares; the metric R cancels for a free D
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--tag", default="global")
    args = ap.parse_args()
    S = Setup(args.layer, args.K, args.tag)
    t0 = time.time()
    K = args.K
    Yc = S.Hc @ S.D.T                                     # original layer output, calibration
    fams = {f["unit"]: f for f in S.families}
    # Fallback "what" for units with no verified family: the old neuron pole that best matches the trained unit.
    aT = hidden(S.Xc, S.P[0], S.P[1])
    Hz = (S.Hc - S.Hc.mean(0)) / (S.Hc.std(0) + 1e-9)
    best_old = []
    for t in range(K):
        a = (aT[:, t] - aT[:, t].mean()) / (aT[:, t].std() + 1e-9)
        c = Hz.T @ a / len(a)
        j = int(np.argmax(np.abs(c)))
        best_old.append((j, float(np.sign(c[j]))))

    def kids_layer(rule):
        G, U = np.zeros((K, 576)), np.zeros((K, 576))
        for t in range(K):
            if t in fams:
                f = fams[t]
                if rule == "PROBE":
                    y = union_labels(f)
                    g = probe_ridge(S.Xc, np.where(y, 1.0, -1.0), np.where(y, 0.5 / y.mean(), 0.5 / (1 - y.mean())))
                    g = g * 2 / (np.std(S.Xc @ g) + 1e-9)
                    u, _ = fit_value_out(S.Xc, g, f["Y_cal"])
                else:
                    out = run_rules(S.Xc, S.Xh, f["Gc"], f["Uc"], f["Dc"], 1, rules=[rule])
                    P = out[rule][0]
                    g, u = P[0][0], P[1][0]
            else:
                j, s = best_old[t]
                g, u = S.G[j], s * S.U[j]
            G[t], U[t] = g, u
        return G, U

    layers = {}
    imp = np.abs(S.Hc).mean(0) * np.linalg.norm(S.D, axis=0)
    keep = np.sort(np.argsort(-imp)[:K])
    layers["CUT"] = (S.G[keep], S.U[keep])
    for rule, name in (("H0", "KIDS-H0"), ("H1", "KIDS-AVG"), ("H6", "KIDS-H6"), ("PROBE", "KIDS-PROBE")):
        layers[name] = kids_layer(rule)
        print(f"built {name} [{time.time() - t0:.0f}s]", flush=True)
    layers["TRAINED-GU"] = (S.P[0], S.P[1])

    orig = S.model.model.layers[S.L].mlp
    held = S.held_blocks
    full = __import__("tree_lib").token_blocks(S.tok, "test", 100)
    ce_base = ce_blocks(S.model, full).mean()
    res = {"layer": S.L, "K": K, "base_ce": float(ce_base), "families": len(fams), "layers": {}}

    def evaluate(G, U, D):
        S.model.model.layers[S.L].mlp = SmallMLP((G, U, D))
        ce = ce_blocks(S.model, full)
        S.model.model.layers[S.L].mlp = orig
        return ce

    ce_tr = evaluate(*S.P)
    res["layers"]["TRAINED"] = {"ce": float(ce_tr.mean())}
    for name, (G, U) in layers.items():
        D = joint_out(S.Xc, G, U, Yc)
        ce = evaluate(G, U, D)
        res["layers"][name] = {"ce": float(ce.mean()), "gap_to_trained": float(ce.mean() - ce_tr.mean())}
        print(f"{name:12s} CE {ce.mean():.4f}  (+{ce.mean() - ce_base:.4f} over the original, "
              f"{ce.mean() - ce_tr.mean():+.4f} vs trained) [{time.time() - t0:.0f}s]", flush=True)
    print(f"TRAINED      CE {ce_tr.mean():.4f}  (+{ce_tr.mean() - ce_base:.4f})")
    json.dump(res, open(os.path.join(ROOT, "results", f"clean-r2round3-L{S.L}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
