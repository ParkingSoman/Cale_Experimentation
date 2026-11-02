"""W1: concrete account of RCF's width effect (CHILDREN-LOG "W1"). Single layer, ORACLE placement (route writes after
MLP L, so the fold's disturbance is excluded; it is measured separately in Q1).

Per (layer, K), all on the dev blocks used for scoring:
  e_c = cut error, e_r = RCF error, rho = e_c - e_r (the repair).
  s_loc = 1 - E|e_r|^2 / E|e_c|^2;  kappa(e) = dCE(e) / E|e|^2 (loss per unit error energy).
  Identity: s_loss = 1 - (kappa_r / kappa_c) (1 - s_loc).
Commonality split of the REMOVED output r (calibration fit, dev measure): R^2 from kept hiddens only, routes only,
both. Unique-to-routes B, shared C, unique-to-kept A, neither N. The cut takes A + C; RCF can add only B.
Injections into the original model: cut (e_c), rcf (e_r), repair alone (rho), half cut (quadratic check).

  python3 research/clean/gate/w1.py --layers 16,22 --Ks 48,192,384,768,1152,1344
"""
import argparse, json, os, sys, time
import numpy as np
import mlx.core as mx, mlx.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from a0 import grams, cv_ridge_g, solve, LAMS, FORBID, ROOT  # noqa: E402
from q1 import setup  # noqa: E402
from a1 import Stash, ZStash  # noqa: E402
from tree_lib import load_model  # noqa: E402
from r1 import block_ce  # noqa: E402
from g6 import boot  # noqa: E402
from g12 import dev_sets, mixed_cal  # noqa: E402

f32 = lambda a: mx.array(np.asarray(a, dtype=np.float32))


class Mix(nn.Module):
    """orig + a (cut - orig) + b (rcf - cut); original output at block position 0."""
    def __init__(self, orig, Gk, Uk, Wc, Wk, Cp, Ca, box):
        super().__init__(); self.orig, self.box = orig, box
        self.Gk, self.Uk, self.Wc, self.Wk, self.Cp, self.Ca = f32(Gk), f32(Uk), f32(Wc), f32(Wk), f32(Cp), f32(Ca); self.a = self.b = 0.0
    def __call__(self, x):
        y0 = self.orig(x); hk = nn.silu(x @ self.Gk.T) * (x @ self.Uk.T)
        cut = hk @ self.Wc; rcf = hk @ self.Wk + self.box["p"] @ self.Cp + self.box["z"] @ self.Ca
        y = y0 + self.a * (cut - y0) + self.b * (rcf - cut)
        return mx.concatenate([y0[:, :1], y[:, 1:]], axis=1)


def r2(Y, P):
    return float(1 - np.sum((Y - P) ** 2) / np.sum((Y - Y.mean(0)) ** 2))


def fitpred(Atr, Ytr, Ate, lams=LAMS):
    W, _ = cv_ridge_g(grams(Atr, Ytr), lams); return Ate @ W.astype(np.float32), W


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="16,22"); ap.add_argument("--Ks", default="48,192,384,768,1152,1344")
    ap.add_argument("--ncal", type=int, default=128)
    a = ap.parse_args(); layers = [int(x) for x in a.layers.split(",")]; assert not FORBID & set(layers)
    model, tok = load_model(); cal = mixed_cal(tok, a.ncal); dev, devc = dev_sets(tok); devall = mx.concatenate([dev, devc], 0)
    texts = (("prose", dev), ("code", devc)); base = {t: block_ce(model, b) for t, b in texts}
    path = os.path.join(ROOT, "results", "clean-w1.json"); out = json.load(open(path)) if os.path.exists(path) else {}; t0 = time.time()
    for L in layers:
        s = setup(model, L, cal); sd = setup(model, L, devall)          # calibration (fit) and dev (measure) rows
        R, Rd = np.concatenate([s["Hp"], s["Z"]], 1), np.concatenate([sd["Hp"], sd["Z"]], 1); P = s["Hp"].shape[1]
        G, U, D = s["W"]; Df = D.astype(np.float32); order = np.argsort(-s["en"]); nd = dev.shape[0] * 511
        lay, prev = model.model.layers[L], model.model.layers[L - 1]; saved = (lay.mlp, lay.self_attn.o_proj, prev.mlp)
        for K in [int(k) for k in a.Ks.split(",")]:
            keep, rem = np.sort(order[:K]), np.sort(order[K:])
            Hk, Hkd = s["H"][:, keep], sd["H"][:, keep]
            r, rd = s["H"][:, rem] @ Df[:, rem].T, sd["H"][:, rem] @ Df[:, rem].T
            # commonality split of the removed output (fit on cal, measured on dev)
            pk, _ = fitpred(Hk, r, Hkd, (1e-4,)); pr, _ = fitpred(R, r, Rd); pkr, _ = fitpred(np.concatenate([Hk, R], 1), r, np.concatenate([Hkd, Rd], 1))
            Rk, Rr, Rkr = r2(rd, pk), r2(rd, pr), r2(rd, pkr)
            split = {"kept_only": Rkr - Rr, "routes_only": Rkr - Rk, "shared": Rk + Rr - Rkr, "neither": 1 - Rkr, "removed_energy_share": float(np.sum(rd ** 2) / np.sum(sd["Y"] ** 2))}
            # cut and RCF fits (as q1), errors on dev rows
            Wc, _ = cv_ridge_g(grams(Hk, s["Y"]), (1e-4,)); W, _ = cv_ridge_g(grams(np.concatenate([Hk, R], 1), s["Y"]), LAMS)
            ec = sd["Y"] - Hkd @ Wc.astype(np.float32); er = sd["Y"] - np.concatenate([Hkd, Rd], 1) @ W.astype(np.float32); rho = ec - er
            E = {t: {k: float(np.mean(np.sum(v[sl] ** 2, 1))) for k, v in (("c", ec), ("r", er), ("rho", rho))}
                 for t, sl in (("prose", slice(0, nd)), ("code", slice(nd, None)))}
            # injections
            box = {}; prev.mlp = Stash(saved[2], *s["Wp"][:2], box, "p"); lay.self_attn.o_proj = ZStash(saved[1], box, "z")
            mix = Mix(saved[0], G[keep], U[keep], Wc, W[:K], W[K:K + P], W[K + P:], box); lay.mlp = mix
            dce = {}
            for nm, (ca, cb) in {"cut": (1, 0), "rcf": (1, 1), "rho": (0, 1), "half_cut": (0.5, 0)}.items():
                mix.a, mix.b = ca, cb
                if nm == "rho":    # orig - rho: inject the repair alone as an error
                    mix.b = -1.0
                dce[nm] = {t: block_ce(model, b) - base[t] for t, b in texts}
            lay.mlp, lay.self_attn.o_proj, prev.mlp = saved
            row = {"split": split}
            for t, _ in texts:
                q = {k: float(v[t].mean()) for k, v in dce.items()}
                s_loc = 1 - E[t]["r"] / E[t]["c"]; kc, kr, krho = q["cut"] / E[t]["c"], q["rcf"] / E[t]["r"], q["rho"] / E[t]["rho"]
                row[t] = {"dce": q, "energy": E[t], "s_loc": s_loc, "s_loss": 1 - q["rcf"] / q["cut"], "kappa_c": kc, "kappa_r": kr, "kappa_rho": krho,
                          "kappa_ratio": kr / kc, "identity_check": 1 - (kr / kc) * (1 - s_loc), "half_cut_ratio": q["half_cut"] / q["cut"],
                          "cross": (q["cut"] + q["rho"] - q["rcf"]) / 2}
            out[f"L{L}_K{K}"] = row; json.dump(out, open(path, "w"), indent=1)
            print(f"L{L} K{K}: split kept {split['kept_only']:.3f} routes {split['routes_only']:.3f} shared {split['shared']:.3f} neither {split['neither']:.3f}"
                  f" (removed energy {split['removed_energy_share']:.2f}) [{time.time()-t0:.0f}s]", flush=True)
            for t, _ in texts:
                x = row[t]
                print(f"   {t}: s_loss {x['s_loss']:.0%} = s_loc {x['s_loc']:.0%} with kappa_r/kappa_c {x['kappa_ratio']:.2f} | kappa_rho/kappa_c {x['kappa_rho']/x['kappa_c']:.2f}"
                      f" | half-cut ratio {x['half_cut_ratio']:.2f} | cross {x['cross']:.4f}", flush=True)
        del s, sd


if __name__ == "__main__":
    main()
