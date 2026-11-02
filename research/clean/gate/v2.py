"""V2: the efficiency ladder (CHILDREN-LOG V2). Single layer L, fold placement, ZERO extra stored numbers in all rungs.
Every rung starts from the closed-form RCF (rung A) and fits with a 3-layer window: the residual after layer L+3 must
match the original model, starting from the input of layer L-1 (no drift: single-layer setting).
  B  free = RCF's own weights (kept writes at L, L-1 write matrix, L o_proj), 15 s
  C  same free weights, 300 s (checkpoint at 150 s)                       B < C  -> our fast fit leaves gains
  D  free = everything in layers L-1 and L except norms (attention q k v o, MLP gate up write, kept G U D at L),
     300 s (checkpoint at 150 s)                                          C < D  -> the fold class is too narrow
Score: share of the closed-form cut's damage.

  python3 research/clean/gate/v2.py --L 16 --Ks 48,768
"""
import argparse, json, os, sys, time
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm.models.base import create_attention_mask
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from a0 import grams, cv_ridge_g, LAMS, ROOT  # noqa: E402
from q1 import setup  # noqa: E402
from a2 import grab_blocks  # noqa: E402
from contract import SwiGLULayer  # noqa: E402
from g10 import Lin  # noqa: E402
from sink import SinkGuard  # noqa: E402
from tree_lib import load_model  # noqa: E402
from r1 import block_ce  # noqa: E402
from g12 import dev_sets, mixed_cal  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--L", type=int, default=16); ap.add_argument("--Ks", default="48,768")
    ap.add_argument("--window", type=int, default=3); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--secs", type=float, default=300); ap.add_argument("--rungs", default="B,C,D"); ap.add_argument("--out", default="clean-v2.json")
    a = ap.parse_args(); L = a.L
    model, tok = load_model(); cal = mixed_cal(tok, 128); dev, devc = dev_sets(tok)
    fitb = mx.concatenate([cal[:64], cal[128:192]], 0); valb = mx.concatenate([cal[64:80], cal[192:208]], 0)
    texts = (("prose", dev), ("code", devc)); base = {t: block_ce(model, b) for t, b in texts}
    m = model.model.layers; up = min(L + a.window, len(m) - 1)
    hin, _ = grab_blocks(model, fitb, L - 1, None)                      # input of layer L-1 (original model)
    hval, _ = grab_blocks(model, valb, L - 1, None)
    mask = create_attention_mask(mx.array(hin[:1]), None)

    def thru(h):
        for l in range(L - 1, up + 1):
            h = m[l](h, mask)
        return h
    tg = np.concatenate([np.array(thru(mx.array(hin[j:j + 4]))) for j in range(0, hin.shape[0], 4)], 0)
    tval = np.concatenate([np.array(thru(mx.array(hval[j:j + 4]))) for j in range(0, hval.shape[0], 4)], 0)

    def val_loss():
        tot = 0.0
        for j in range(0, hval.shape[0], 8):
            t = mx.array(tval[j:j + 8]); r = (thru(mx.array(hval[j:j + 8])) - t)[:, 1:]; c = t[:, 1:] - t[:, 1:].mean(-1, keepdims=True)
            tot += float((r * r).sum() / (c * c).sum())
        return tot
    s = setup(model, L, cal); G, U, D = s["W"]; Gp, Up, Dp = s["Wp"]; P = s["Hp"].shape[1]
    R = np.concatenate([s["Hp"], s["Z"]], 1); order = np.argsort(-s["en"])
    lay, prev = m[L], m[L - 1]
    saved = {"mlp": lay.mlp, "o": lay.self_attn.o_proj, "pmlp": prev.mlp,
             "attn": {l: {k: np.array(getattr(m[l].self_attn, k).weight) for k in ("q_proj", "k_proj", "v_proj")} for l in (L - 1, L)},
             "po": np.array(prev.self_attn.o_proj.weight)}
    Wo = np.array(saved["o"].weight, dtype=np.float64)
    path = os.path.join(ROOT, "results", a.out); out = json.load(open(path)) if os.path.exists(path) else {}; t0 = time.time()

    def restore():
        lay.mlp, lay.self_attn.o_proj, prev.mlp = saved["mlp"], saved["o"], saved["pmlp"]
        for l in (L - 1, L):
            for k, w in saved["attn"][l].items():
                getattr(m[l].self_attn, k).weight = mx.array(w)
        prev.self_attn.o_proj.weight = mx.array(saved["po"])

    def install(keep, Dk, Cp=None, Ca=None):
        lay.mlp = SinkGuard(SwiGLULayer(G[keep], U[keep], Dk), SwiGLULayer(G, U, D))
        if Cp is not None:
            prev.mlp = SinkGuard(SwiGLULayer(Gp, Up, Dp + Cp.T), SwiGLULayer(Gp, Up, Dp))
            lay.self_attn.o_proj = SinkGuard(Lin(Wo + Ca.T), Lin(Wo))

    def score():
        return {t: float((block_ce(model, b) - base[t]).mean()) for t, b in texts}

    def unfreeze_set(rung):
        lay.mlp.new.unfreeze(keys=["D"]); prev.mlp.new.unfreeze(keys=["D"]); lay.self_attn.o_proj.new.unfreeze()
        if rung in ("D", "Dgate", "G", "Gcut", "Ges"):
            lay.mlp.new.unfreeze()
        if rung in ("D", "Dprev", "G", "Gcut", "Ges"):
            prev.mlp.new.unfreeze()
        if rung in ("D", "Dattn"):
            prev.self_attn.o_proj.unfreeze()
            for l in (L - 1, L):
                for k in ("q_proj", "k_proj", "v_proj"):
                    getattr(m[l].self_attn, k).unfreeze()

    def train(rung, secs_list):
        model.freeze(); unfreeze_set(rung)

        def loss_fn(mod, h, t):
            r = (thru(h) - t)[:, 1:]; c = t[:, 1:] - t[:, 1:].mean(-1, keepdims=True)
            return (r * r).sum() / (c * c).sum()
        vg = nn.value_and_grad(model, loss_fn); opt = optim.Adam(learning_rate=a.lr); rng = np.random.default_rng(L)
        res, ts, steps, done = {}, time.time(), 0, 0.0
        best = (val_loss(), dict(tree_flatten(model.trainable_parameters())), 0) if rung == "Ges" else None
        for secs in secs_list:
            while time.time() - ts < secs:
                idx = rng.integers(0, hin.shape[0], 4)
                lv, g = vg(model, mx.array(hin[idx]), mx.array(tg[idx])); opt.update(model, g); mx.eval(model.parameters(), opt.state); steps += 1
                if best is not None and steps % 25 == 0:
                    v = val_loss()
                    if v < best[0]:
                        best = (v, {k: mx.array(w) for k, w in tree_flatten(model.trainable_parameters())}, steps)
            if best is not None:
                model.load_weights(list(best[1].items()), strict=False); res["Ges_best_step"] = best[2]
            model.unfreeze(); res[f"{rung}_{int(secs)}s"] = {**score(), "steps": steps, "loss": float(lv)}; model.freeze(); unfreeze_set(rung)
        model.unfreeze()
        return res

    for K in [int(k) for k in a.Ks.split(",")]:
        keep = np.sort(order[:K]); Hk = s["H"][:, keep]
        Wc, _ = cv_ridge_g(grams(Hk, s["Y"]), (1e-4,)); W, _ = cv_ridge_g(grams(np.concatenate([Hk, R], 1), s["Y"]), LAMS)
        rows = {}
        install(keep, Wc.T); rows["cut"] = score(); restore()
        install(keep, W[:K].T, W[K:K + P], W[K + P:]); rows["A"] = score(); restore()
        plan = {"B": [15], "C": [a.secs / 2, a.secs], "D": [a.secs / 2, a.secs], "Dgate": [a.secs], "Dprev": [a.secs], "Dattn": [a.secs], "G": [a.secs / 2, a.secs], "Gcut": [a.secs / 2, a.secs], "Ges": [a.secs]}
        for rung, secs in ((r, plan[r]) for r in a.rungs.split(",")):
            if rung == "Gcut":   # same free weights as G, but start from the cut (folds start at zero)
                install(keep, Wc.T, np.zeros((P, D.shape[0])), np.zeros((D.shape[0], D.shape[0])))
            else:
                install(keep, W[:K].T, W[K:K + P], W[K + P:])
            rows.update(train(rung, secs)); restore()
            print(f"L{L} K{K} rung {rung} done [{time.time()-t0:.0f}s]", flush=True)
        row = {"raw": rows}
        for nm, v in rows.items():
            if nm != "cut" and isinstance(v, dict):
                row[nm] = {t: 1 - v[t] / rows["cut"][t] for t, _ in texts}
        out[f"L{L}_K{K}"] = row; json.dump(out, open(path, "w"), indent=1)
        print(f"L{L} K{K} share of cut damage (prose / code): " + " | ".join(f"{nm} {v['prose']:.0%}/{v['code']:.0%}" for nm, v in row.items() if nm != "raw"), flush=True)


if __name__ == "__main__":
    main()
