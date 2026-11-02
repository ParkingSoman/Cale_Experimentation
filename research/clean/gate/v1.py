"""V1: per-neuron audit of the "neither" bucket (CHILDREN-LOG V1). Single layer, oracle placement.
Features F = [kept hiddens, L-1 hiddens, L attention outputs] (RCF today). The leftover splits EXACTLY per removed
neuron: e = sum_j (h_j - F w_j) d_j (least squares is linear in the target). For each removed neuron:
  pre_r2 = min(R^2 of its gate pre-activation, R^2 of its up pre-activation) from F   (can the routes read its input?)
  out_r2 = R^2 of its output h_j from F
Groups: BEND (pre_r2 >= .8: input reachable, output not) and MISSING (pre_r2 < .8: input not reachable).
Each group's leftover is injected alone into the original model to get its loss.

  python3 research/clean/gate/v1.py --L 16 --Ks 48,768
"""
import argparse, json, os, sys, time
import numpy as np
import mlx.core as mx, mlx.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from a0 import grams, solve, LAMS, ROOT  # noqa: E402
from q1 import setup  # noqa: E402
from a1 import Stash, ZStash  # noqa: E402
from tree_lib import load_model  # noqa: E402
from r1 import block_ce  # noqa: E402
from g12 import dev_sets, mixed_cal  # noqa: E402

f32 = lambda a: mx.array(np.asarray(a, dtype=np.float32))


class Leftover(nn.Module):
    """orig output minus the leftover of neuron set S: sum_{j in S} (h_j - F w_j) d_j. Position 0 untouched."""
    def __init__(self, orig, Gk, Uk, GS, US, DS, M, box):
        super().__init__(); self.orig, self.box = orig, box
        self.Gk, self.Uk, self.GS, self.US, self.DS, self.M = f32(Gk), f32(Uk), f32(GS), f32(US), f32(DS), f32(M)
    def __call__(self, x):
        y0 = self.orig(x)
        F = mx.concatenate([nn.silu(x @ self.Gk.T) * (x @ self.Uk.T), self.box["p"], self.box["z"]], axis=-1)
        e = (nn.silu(x @ self.GS.T) * (x @ self.US.T)) @ self.DS.T - F @ self.M
        y = y0 - e
        return mx.concatenate([y0[:, :1], y[:, 1:]], axis=1)


def r2cols(Y, P):
    return 1 - np.sum((Y - P) ** 2, 0) / np.maximum(np.sum((Y - Y.mean(0)) ** 2, 0), 1e-12)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--L", type=int, default=16); ap.add_argument("--Ks", default="48,768"); ap.add_argument("--lam", type=float, default=1e-2)
    a = ap.parse_args(); L = a.L
    model, tok = load_model(); cal = mixed_cal(tok, 128); dev, devc = dev_sets(tok); devall = mx.concatenate([dev, devc], 0)
    texts = (("prose", dev), ("code", devc)); base = {t: block_ce(model, b) for t, b in texts}
    s, sd = setup(model, L, cal), setup(model, L, devall); G, U, D = (w.astype(np.float32) for w in s["W"])
    lay, prev = model.model.layers[L], model.model.layers[L - 1]; saved = (lay.mlp, lay.self_attn.o_proj, prev.mlp)
    path = os.path.join(ROOT, "results", "clean-v1.json"); out = json.load(open(path)) if os.path.exists(path) else {}; t0 = time.time()
    order = np.argsort(-s["en"])
    for K in [int(k) for k in a.Ks.split(",")]:
        keep, rem = np.sort(order[:K]), np.sort(order[K:])
        F = np.concatenate([s["H"][:, keep], s["Hp"], s["Z"]], 1); Fd = np.concatenate([sd["H"][:, keep], sd["Hp"], sd["Z"]], 1)
        Gm = (F.T @ F).astype(np.float64)
        def fit(Y):
            W = solve(Gm, (F.T @ Y).astype(np.float64), a.lam).astype(np.float32); return W
        Xc, Xd = s["X"], sd["X"]
        Wg = fit(Xc @ G[rem].T); Wu = fit(Xc @ U[rem].T); Wh = fit(s["H"][:, rem])
        r2g = r2cols(Xd @ G[rem].T, Fd @ Wg); r2u = r2cols(Xd @ U[rem].T, Fd @ Wu); r2h = r2cols(sd["H"][:, rem], Fd @ Wh)
        pre = np.minimum(r2g, r2u)
        # unreached energy per neuron on dev rows (its own leftover times its write norm); exact total via the sum
        E = sd["H"][:, rem] - Fd @ Wh; ue = np.sum(E ** 2, 0) * np.sum(D[:, rem] ** 2, 0)
        groups = {"all": np.ones(len(rem), bool), "bend": pre >= 0.8, "missing": pre < 0.8}
        row = {"n_removed": int(len(rem)), "pre_r2_median": float(np.median(pre)), "out_r2_median": float(np.median(r2h)),
               "pre_r2_quantiles": [float(q) for q in np.quantile(pre, [.1, .25, .5, .75, .9])]}
        box = {}; prev.mlp = Stash(saved[2], *s["Wp"][:2], box, "p"); lay.self_attn.o_proj = ZStash(saved[1], box, "z")
        for g, m in groups.items():
            S = rem[m]; M = Wh[:, m] @ D[:, S].T
            lay.mlp = Leftover(saved[0], G[keep], U[keep], G[S], U[S], D[:, S], M, box)
            d = {t: float((block_ce(model, b) - base[t]).mean()) for t, b in texts}
            row[g] = {"n": int(m.sum()), "energy_share": float(ue[m].sum() / ue.sum()), "dce": d,
                      "out_r2_median": float(np.median(r2h[m])) if m.any() else None}
        lay.mlp, lay.self_attn.o_proj, prev.mlp = saved
        out[f"L{L}_K{K}"] = row; json.dump(out, open(path, "w"), indent=1)
        print(f"L{L} K{K}: input reading R^2 median {row['pre_r2_median']:.2f} (q10 {row['pre_r2_quantiles'][0]:.2f}), output R^2 median {row['out_r2_median']:.2f}", flush=True)
        for g in groups:
            x = row[g]; print(f"   {g:8s} n {x['n']:5d} energy {x['energy_share']:.0%} dCE prose {x['dce']['prose']:.4f} code {x['dce']['code']:.4f}", flush=True)
        print(f"   [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
