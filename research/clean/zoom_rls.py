"""Blue sky #9: runtime closed-form refit from stored statistics, instead of learned output corrections.

A saved zoom configuration (zoom1.py --save NAME) is evaluated in two ways at every dial setting:
  stored   as built: originals use D + beta*dD + alpha_own*eD (the learned corrections)
  rls      the learned corrections are dropped. The originals' output weights are solved in closed form:
           D_on = argmin || (Y - parents' faded output) - H_on D_on^T ||  on calibration text.
           The runtime needs only two stored tables: A = H^T H (1536 x 1536) and B = H^T (Y - ...) pieces,
           so the solve takes milliseconds when the dial moves, not per token. No training at runtime.
Loss through the 2 layers after layer L on cached inputs, on the chosen block range.

  python3 research/clean/zoom_rls.py --config v14 --lo 20 --hi 40
Writes results/clean-zoomrls-<config>.json.
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "research", "tree"))
sys.path.insert(0, HERE)
from tree_lib import load_model, token_blocks, Weights, hidden, SEQ  # noqa: E402
from r1 import capture, SmallMLP  # noqa: E402
from r2round5 import cache_h, tail  # noqa: E402
from r2round3 import joint_out  # noqa: E402

TAU = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lo", type=int, default=20)
    ap.add_argument("--hi", type=int, default=40)
    ap.add_argument("--n_cal", type=int, default=32)
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--tau", type=float, default=0.0)
    args = ap.parse_args()
    global TAU
    TAU = args.tau
    t0 = time.time()
    z = np.load(os.path.join(HERE, "r1_weights", f"zoom_{args.config}.npz"))
    cfg = {k: z[k] for k in z.files}
    resid = [r[~np.isnan(r)] for r in cfg["resid"]]
    L, K = int(cfg["layer"]), int(cfg["K"])
    fam, order, imp = cfg["fam"], cfg["order"], cfg["imp"]
    model, tok = load_model()
    W = Weights(model)
    G, U, D = [a.astype(np.float64) for a in (W.G[L], W.U[L], W.D[L])]
    n = G.shape[0]
    S = types.SimpleNamespace(model=model, L=L)
    cal = token_blocks(tok, "train", args.n_cal)
    held = token_blocks(tok, "test", 100)[args.lo:args.hi]
    Hheld = mx.array(cache_h(S, held))
    labels = np.array(held[:, 1:])
    Xc = capture(model, cal, [L])[L]
    Xc = Xc[np.arange(len(Xc)) % SEQ != 0].astype(np.float64)
    Hc = hidden(Xc, G, U)
    Yc = Hc @ D.T
    Pc = hidden(Xc, cfg["G"], cfg["U"])                    # parents' activations on calibration text
    # The two stored tables (computed once, offline).
    A = Hc.T @ Hc
    BY = Hc.T @ Yc                                         # n x 576
    BP = Hc.T @ Pc                                         # n x K   (to subtract the parents' faded output)
    fam_size = np.bincount(fam, minlength=K)

    def ce(P):
        mlp = SmallMLP(P)
        tot = 0.0
        for i in range(0, Hheld.shape[0], 4):
            lg = tail(S, Hheld[i:i + 4], mlp)
            lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            nll = -mx.take_along_axis(lg, mx.array(labels[i:i + 4])[..., None], axis=-1).squeeze(-1)[:, 1:]
            tot += float(nll.sum().item())
        return tot / (labels.shape[0] * (labels.shape[1] - 1))

    base = ce((G, U, D))

    def state(N):
        best = None
        for M in range(n + 1):
            on = np.zeros(n, bool)
            on[order[:M]] = True
            k_on = np.bincount(fam[on], minlength=K)
            alpha = np.array([resid[t][k_on[t]] if k_on[t] < fam_size[t] else 0.0 for t in range(K)])
            alpha = np.where(alpha < TAU, 0.0, alpha)          # a parent with a small missing share frees its slot
            if M + int((alpha > 1e-9).sum()) <= N:
                best = (on, alpha)
            else:
                break
        return best

    def layer(N, mode):
        on, alpha = state(N)
        keep_p = np.where(alpha > 1e-9)[0]
        idx = np.where(on)[0]
        Dp = cfg["D"][:, keep_p] * alpha[keep_p][None, :]
        if mode == "stored":
            beta = float(1 - imp[on].sum() / imp.sum())
            if "bk" in cfg and len(cfg["bk"]) > 0:
                beta = float(np.interp(on.mean(), np.linspace(0, 1, len(cfg["bk"])), cfg["bk"])) * (1 - on.mean())
            Do = D[:, idx] + beta * cfg["dD"][:, idx] + cfg["eD"][:, idx] * alpha[fam[idx]][None, :]
        else:
            # Closed form: originals fit what the layer output still needs after the parents' faded output.
            if len(idx):
                Aon = A[np.ix_(idx, idx)]
                Aon = Aon + np.eye(len(idx)) * args.ridge * np.trace(Aon) / len(idx)
                rhs = BY[idx] - BP[idx][:, keep_p] @ Dp.T          # H_on^T (Y - P_keep Dp^T)
                Do = np.linalg.solve(Aon, rhs).T
            else:
                Do = np.zeros((576, 0))
        Gl = np.concatenate([cfg["G"][keep_p], G[idx]])
        Ul = np.concatenate([cfg["U"][keep_p], U[idx]])
        return Gl, Ul, np.concatenate([Dp, Do], 1)

    imp_cut = np.abs(Hc).mean(0) * np.linalg.norm(D, axis=0)

    def cut(N):
        keep = np.sort(np.argsort(-imp_cut)[:N])
        return G[keep], U[keep], joint_out(Xc, G[keep], U[keep], Yc)

    sweep = sorted(set([K] + list(range(64, n, 32)) + [2 * K, 4 * K, 8 * K, 16 * K, 32 * K] + [n]))
    res = {"config": args.config, "blocks": [args.lo, args.hi], "base": base, "rows": []}
    for N in sweep:
        r = {"N": N, "stored": ce(layer(N, "stored")) - base, "rls": ce(layer(N, "rls")) - base}
        if N in (K, 2 * K, 4 * K, 8 * K, 16 * K, 32 * K):
            r["cut"] = ce(cut(N)) - base
        res["rows"].append(r)
    for mode in ("stored", "rls"):
        v = [r[mode] for r in res["rows"]]
        rises = [v[i + 1] - v[i] for i in range(len(v) - 1)]
        res[f"{mode}_max_rise"] = max(rises)
    json.dump(res, open(os.path.join(ROOT, "results", f"clean-zoomrls-{args.config}-tau{args.tau}.json"), "w"), indent=1)
    print(f"tau {args.tau}: {args.config} blocks {args.lo}-{args.hi}  size: stored | runtime-LS | cut")
    for r in res["rows"]:
        if "cut" in r or r["N"] == n:
            print(f"{r['N']:5d}  +{r['stored']:.4f} | +{r['rls']:.4f} | " + (f"+{r['cut']:.4f}" if "cut" in r else ""))
    print(f"max rise along the dial: stored {res['stored_max_rise']:.4f}, runtime-LS {res['rls_max_rise']:.4f} "
          f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
