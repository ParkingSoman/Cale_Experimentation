"""Independent final check of a saved zoom configuration (built by zoom1.py --save NAME).

Independent of the development loop in four ways:
  data    prose test blocks 60-99 (never used while iterating; iteration used blocks 0-19) and Python code test text
  model   loss through the FULL model (all 30 layers), not the 2-layer shortcut on cached inputs
  code    the dial is rebuilt here from the saved file with fresh code (no import of zoom1's dial logic)
  refs    the cut (LS refit) and per-size fitted layers are rebuilt here
Criteria (as fixed before the runs): 1 steady (strict 1e-4 and practical 1e-3 reported), 2 beats the cut at every
test size, 3 full setting = original, 4 within 25% of a per-size fitted layer.

  python3 research/clean/zoomfinal.py --config v10 [--ref_secs 330]
Writes results/clean-zoomfinal-<config>.json.
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np
import mlx.core as mx
import mlx.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "research", "tree"))
sys.path.insert(0, HERE)
from tree_lib import load_model, token_blocks, Weights, hidden, SEQ  # noqa: E402
from r1 import capture, SmallMLP  # noqa: E402
from r2round4 import text_blocks  # noqa: E402
from r2round5 import cache_h, fit_tail  # noqa: E402
from r2round3 import joint_out  # noqa: E402


class DialLayer(nn.Module):
    """Layer-L MLP at one dial setting, rebuilt from the saved configuration."""

    def __init__(self, cfg, G, U, D, N):
        super().__init__()
        fam, order, imp, resid = cfg["fam"], cfg["order"], cfg["imp"], cfg["resid"]
        K, n = int(cfg["K"]), len(fam)
        fam_size = np.bincount(fam, minlength=K)

        def state(M):
            on = np.zeros(n, bool)
            on[order[:M]] = True
            k_on = np.bincount(fam[on], minlength=K)
            alpha = np.array([resid[t][k_on[t]] if k_on[t] < fam_size[t] else 0.0 for t in range(K)])
            return on, alpha, int(on.sum() + (alpha > 1e-9).sum())

        M = 0
        for m in range(n + 1):                        # plain scan: the largest M whose total fits in N
            if state(m)[2] <= N:
                M = m
        on, alpha, self.neurons = state(M)
        beta = float(1 - imp[on].sum() / imp.sum()) if bool(cfg["corr"]) else 0.0
        if bool(cfg["corr"]) and "bk" in cfg and len(cfg["bk"]) > 0:
            bk = cfg["bk"]                       # learned schedule over the share of originals on
            beta = float(np.interp(on.mean(), np.linspace(0, 1, len(bk)), bk)) * (1 - on.mean())   # 0 at full
        keep_p = np.where(alpha > 1e-9)[0]
        idx = np.where(on)[0]
        # Parents (scaled output) and original neurons (original output + beta x correction), as one plain MLP.
        Gl = np.concatenate([cfg["G"][keep_p], G[idx]])
        Ul = np.concatenate([cfg["U"][keep_p], U[idx]])
        Do = D[:, idx] + beta * cfg["dD"][:, idx]
        if "famcorr" in cfg and bool(cfg["famcorr"]):
            # Second correction: scaled by how much of the neuron's OWN family is still missing (its parent's scale).
            Do = Do + cfg["eD"][:, idx] * alpha[fam[idx]][None, :]
        Dl = np.concatenate([cfg["D"][:, keep_p] * alpha[keep_p][None, :], Do], 1)
        self.mlp = SmallMLP((Gl, Ul, Dl))

    def __call__(self, x):
        return self.mlp(x)


PER_BLOCK = {}


def ce_full(model, blocks, batch=4, key=None):
    """Mean CE (position 0 left out). With key, also keeps the per-block means for a paired bootstrap."""
    per = []
    for i in range(0, blocks.shape[0], batch):
        b = blocks[i:i + batch]
        lg = model(b[:, :-1])
        lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        nll = -mx.take_along_axis(lg, b[:, 1:, None], axis=-1).squeeze(-1)[:, 1:]
        per.append(np.array(nll.mean(1)))
    per = np.concatenate(per)
    if key is not None:
        PER_BLOCK[key] = per
    return float(per.mean())


def boot_margin(a, b, n=2000, seed=0):
    """Paired bootstrap over blocks of mean(b - a). Returns (mean, 2.5%, 97.5%)."""
    d = PER_BLOCK[b] - PER_BLOCK[a]
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ref_secs", type=float, default=330)
    ap.add_argument("--sweep_step", type=int, default=32)
    ap.add_argument("--final_lo", type=int, default=40, help="final prose blocks: 40-59 unused so far (60-99 used once)")
    ap.add_argument("--final_hi", type=int, default=60)
    ap.add_argument("--tag", default="", help="suffix for the result file")
    args = ap.parse_args()
    t0 = time.time()
    z = np.load(os.path.join(HERE, "r1_weights", f"zoom_{args.config}.npz"))
    cfg = {k: z[k] for k in z.files}
    cfg["resid"] = [r[~np.isnan(r)] for r in cfg["resid"]]
    L, K = int(cfg["layer"]), int(cfg["K"])
    model, tok = load_model()
    W = Weights(model)
    G, U, D = [a.astype(np.float64) for a in (W.G[L], W.U[L], W.D[L])]
    n = G.shape[0]
    prose = token_blocks(tok, "test", 100)[args.final_lo:args.final_hi]
    code = text_blocks(tok, os.path.join(HERE, "data", "code_test.txt"), 40)
    orig = model.model.layers[L].mlp
    base = {"prose": ce_full(model, prose), "code": ce_full(model, code)}

    def ce_with(mlp, which=("prose", "code"), tag=None):
        model.model.layers[L].mlp = mlp
        r = {k: ce_full(model, prose if k == "prose" else code, key=(tag, k) if tag else None) - base[k]
             for k in which}
        model.model.layers[L].mlp = orig
        return r

    # Calibration data for the references (prose train blocks, as in development).
    cal = token_blocks(tok, "train", 32)
    Xc = capture(model, cal, [L])[L]
    Xc = Xc[np.arange(len(Xc)) % SEQ != 0].astype(np.float64)
    Hc = hidden(Xc, G, U)
    Yc = Hc @ D.T
    imp_cut = np.abs(Hc).mean(0) * np.linalg.norm(D, axis=0)

    def cut(N):
        keep = np.sort(np.argsort(-imp_cut)[:N])
        return G[keep], U[keep], joint_out(Xc, G[keep], U[keep], Yc)

    full = DialLayer(cfg, G, U, D, n)
    pc = {"original_layer_params": int(3 * 576 * n), "parents": int(3 * 576 * K),
          "corrections": int(576 * n * ((1 if bool(cfg["corr"]) else 0) + (1 if "famcorr" in cfg and bool(cfg["famcorr"]) else 0)))
          if int(cfg.get("corr_rank", 0)) == 0 else
          int((576 + n) * int(cfg["corr_rank"]) * ((1 if bool(cfg["corr"]) else 0) + (1 if "famcorr" in cfg and bool(cfg["famcorr"]) else 0)))}
    res = {"config": args.config, "layer": L, "K": K, "base": base, "params": pc,
           "full_setting": {"neurons": full.neurons, **ce_with(full)}}
    print(f"base {base} | full setting {res['full_setting']} [{time.time() - t0:.0f}s]", flush=True)
    test_sizes = [N for N in (K, 2 * K, 4 * K, 8 * K, 16 * K, 32 * K) if N < n]      # claim covers sizes below full
    sweep = sorted(N for N in set([K] + list(range(32, n, args.sweep_step)) + test_sizes) if N >= K)
    dial = []
    for N in sweep:
        d = DialLayer(cfg, G, U, D, N)
        dial.append({"N": N, "neurons": d.neurons, **ce_with(d, ("prose",))})
    print("prose sweep:", " ".join(f"{d['neurons']}:{d['prose']:.3f}" for d in dial), flush=True)
    rises = [dial[i + 1]["prose"] - dial[i]["prose"] for i in range(len(dial) - 1)]
    S = types.SimpleNamespace(model=model, L=L)
    Hcal = cache_h(S, cal)
    cmp = []
    for N in test_sizes:
        d = DialLayer(cfg, G, U, D, N)
        row = {"N": N, "neurons": d.neurons, "dial": ce_with(d, tag=("dial", N)),
               "cut": ce_with(SmallMLP(cut(d.neurons)), tag=("cut", N))}
        if args.ref_secs > 0:
            Pf, fsteps = fit_tail(S, Hcal, cut(d.neurons), args.ref_secs / 60)
            row["fitted"] = ce_with(SmallMLP(Pf))
            row["fitted_steps"] = fsteps
        else:                                    # quick mode: no per-size fitted reference
            row["fitted"] = {"prose": float("nan"), "code": float("nan")}
        # Paired bootstrap over blocks: cut minus dial (> 0 means the dial is better), 95% interval.
        row["margin_vs_cut"] = {k: boot_margin((("dial", N), k), (("cut", N), k)) for k in ("prose", "code")}
        cmp.append(row)
        print(f"N {N:4d} ({d.neurons}) dial prose +{row['dial']['prose']:.4f} code +{row['dial']['code']:.4f} | cut "
              f"+{row['cut']['prose']:.4f} / +{row['cut']['code']:.4f} | fitted +{row['fitted']['prose']:.4f} / "
              f"+{row['fitted']['code']:.4f} [{time.time() - t0:.0f}s]", flush=True)
        res["compare"] = cmp
        json.dump(res, open(os.path.join(ROOT, "results", f"clean-zoomfinal-{args.config}{args.tag}.json"), "w"), indent=1)
    res["dial"] = dial
    res["criteria_prose"] = {
        "1_steady_strict": all(r <= 1e-4 for r in rises), "1b_steady_practical": all(r <= 1e-3 for r in rises),
        "max_rise": max(rises),
        "2_beats_cut": all(r["dial"]["prose"] < r["cut"]["prose"] for r in cmp),
        "2b_beats_cut_95pct": all(r["margin_vs_cut"]["prose"][1] > 0 for r in cmp),
        "3_full_equals_original": abs(res["full_setting"]["prose"]) < 1e-3,
        "4_near_fitted": all(r["dial"]["prose"] <= 1.25 * r["fitted"]["prose"] + 1e-4 for r in cmp)}
    res["criteria_code"] = {
        "2_beats_cut": all(r["dial"]["code"] < r["cut"]["code"] for r in cmp),
        "2b_beats_cut_95pct": all(r["margin_vs_cut"]["code"][1] > 0 for r in cmp),
        "3_full_equals_original": abs(res["full_setting"]["code"]) < 1e-3,
        "4_near_fitted": all(r["dial"]["code"] <= 1.25 * r["fitted"]["code"] + 1e-4 for r in cmp)}
    json.dump(res, open(os.path.join(ROOT, "results", f"clean-zoomfinal-{args.config}{args.tag}.json"), "w"), indent=1)
    print("criteria prose:", res["criteria_prose"])
    print("criteria code:", res["criteria_code"], f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
