"""T1: two-back RCF (CHILDREN-LOG T1). Single layer L. Routes = layer L-1 AND layer L-2 MLP hiddens + layer L attention.
Arms (share of the closed-form cut's damage):
  A1   one-back closed form (today's RCF), fold placement
  B1   A1 + 15 s window fit (free: kept writes, L-1 write matrix, L o_proj)
  A2o  two-back closed form, ORACLE placement (all route writes after MLP L; extra storage, upper bound)
  A2   two-back closed form, fold placement (L-2 fold into L-2's write matrix)
  B2   A2 + 15 s window fit (free: kept writes, L-1 and L-2 write matrices, L o_proj)
Window: residual after L+3 vs the original, starting from the input of layer L-2.

  python3 research/clean/gate/t1.py --layers 16,22 --Ks 48,768
"""
import argparse, json, os, sys, time
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx_lm.models.base import create_attention_mask
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from a0 import grams, cv_ridge_g, LAMS, FORBID, ROOT  # noqa: E402
from q1 import setup  # noqa: E402
from a1 import Stash, ZStash, AddAfter  # noqa: E402
from a2 import grab_blocks  # noqa: E402
from contract import SwiGLULayer, orig_weights  # noqa: E402
from g6 import silu  # noqa: E402
from g10 import Lin  # noqa: E402
from g12 import capture, dev_sets, mixed_cal  # noqa: E402
from sink import SinkGuard  # noqa: E402
from tree_lib import load_model  # noqa: E402
from r1 import block_ce  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="16,22"); ap.add_argument("--Ks", default="48,768")
    ap.add_argument("--secs", type=float, default=15); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--window", type=int, default=3)
    a = ap.parse_args(); layers = [int(x) for x in a.layers.split(",")]; assert not FORBID & set(layers)
    model, tok = load_model(); cal = mixed_cal(tok, 128); dev, devc = dev_sets(tok)
    fitb = mx.concatenate([cal[:64], cal[128:192]], 0); texts = (("prose", dev), ("code", devc))
    base = {t: block_ce(model, b) for t, b in texts}; m = model.model.layers
    path = os.path.join(ROOT, "results", "clean-t1.json" if a.secs == 15 else f"clean-t1-{int(a.secs)}s.json"); out = json.load(open(path)) if os.path.exists(path) else {}; t0 = time.time()
    for L in layers:
        up = min(L + a.window, len(m) - 1)
        hin, _ = grab_blocks(model, fitb, L - 2, None); mask = create_attention_mask(mx.array(hin[:1]), None)

        def thru(h):
            for l in range(L - 2, up + 1):
                h = m[l](h, mask)
            return h
        tg = np.concatenate([np.array(thru(mx.array(hin[j:j + 4]))) for j in range(0, hin.shape[0], 4)], 0)
        s = setup(model, L, cal); G, U, D = s["W"]; Gp, Up, Dp = s["Wp"]; P = s["Hp"].shape[1]
        G2, U2, D2 = orig_weights(model, L - 2)
        X2 = capture(model, cal, {"x": (m[L - 2], "mlp")})["x"]
        Hp2 = (silu(X2 @ G2.T.astype(np.float32)) * (X2 @ U2.T.astype(np.float32))).astype(np.float32); del X2
        lay, prev, prev2 = m[L], m[L - 1], m[L - 2]; saved = (lay.mlp, lay.self_attn.o_proj, prev.mlp, prev2.mlp)
        Wo = np.array(saved[1].weight, dtype=np.float64); order = np.argsort(-s["en"])

        def restore():
            lay.mlp, lay.self_attn.o_proj, prev.mlp, prev2.mlp = saved

        def score():
            return {t: float((block_ce(model, b) - base[t]).mean()) for t, b in texts}

        def install(keep, Dk, Cp=None, Ca=None, C2=None, oracle=False):
            mlp = SinkGuard(SwiGLULayer(G[keep], U[keep], Dk), SwiGLULayer(G, U, D))
            if oracle:
                box, parts = {}, []
                prev.mlp = Stash(saved[2], Gp, Up, box, "p"); parts.append(("p", Cp.T))
                prev2.mlp = Stash(saved[3], G2, U2, box, "p2"); parts.append(("p2", C2.T))
                lay.self_attn.o_proj = ZStash(saved[1], box, "z"); parts.append(("z", Ca.T))
                lay.mlp = AddAfter(mlp, parts, box); return
            lay.mlp = mlp
            if Cp is not None:
                prev.mlp = SinkGuard(SwiGLULayer(Gp, Up, Dp + Cp.T), SwiGLULayer(Gp, Up, Dp))
                lay.self_attn.o_proj = SinkGuard(Lin(Wo + Ca.T), Lin(Wo))
            if C2 is not None:
                prev2.mlp = SinkGuard(SwiGLULayer(G2, U2, D2 + C2.T), SwiGLULayer(G2, U2, D2))

        def fit(two):
            model.freeze(); lay.mlp.new.unfreeze(keys=["D"]); prev.mlp.new.unfreeze(keys=["D"]); lay.self_attn.o_proj.new.unfreeze()
            if two:
                prev2.mlp.new.unfreeze(keys=["D"])

            def loss_fn(mod, h, t):
                r = (thru(h) - t)[:, 1:]; c = t[:, 1:] - t[:, 1:].mean(-1, keepdims=True); return (r * r).sum() / (c * c).sum()
            vg = nn.value_and_grad(model, loss_fn); opt = optim.Adam(learning_rate=a.lr); rng = np.random.default_rng(L); ts = time.time(); n = 0
            while time.time() - ts < a.secs:
                idx = rng.integers(0, hin.shape[0], 4)
                _, g = vg(model, mx.array(hin[idx]), mx.array(tg[idx])); opt.update(model, g); mx.eval(model.parameters(), opt.state); n += 1
            model.unfreeze(); return n

        for K in [int(k) for k in a.Ks.split(",")]:
            keep = np.sort(order[:K]); Hk = s["H"][:, keep]; rows = {}
            Wc, _ = cv_ridge_g(grams(Hk, s["Y"]), (1e-4,)); install(keep, Wc.T); rows["cut"] = score(); restore()
            W1, _ = cv_ridge_g(grams(np.concatenate([Hk, s["Hp"], s["Z"]], 1), s["Y"]), LAMS)
            W2, _ = cv_ridge_g(grams(np.concatenate([Hk, s["Hp"], Hp2, s["Z"]], 1), s["Y"]), LAMS)
            a1 = (W1[:K].T, W1[K:K + P], W1[K + P:]); a2 = (W2[:K].T, W2[K:K + P], W2[K + 2 * P:], W2[K + P:K + 2 * P])
            install(keep, *a1); rows["A1"] = score(); restore()
            install(keep, *a1); fit(False); rows["B1"] = score(); restore()
            install(keep, *a2, oracle=True); rows["A2o"] = score(); restore()
            install(keep, *a2); rows["A2"] = score(); restore()
            install(keep, *a2); fit(True); rows["B2"] = score(); restore()
            row = {"raw": rows}
            for nm, v in rows.items():
                if nm != "cut":
                    row[nm] = {t: 1 - v[t] / rows["cut"][t] for t, _ in texts}
            out[f"L{L}_K{K}"] = row; json.dump(out, open(path, "w"), indent=1)
            print(f"L{L} K{K} share (prose / code): " + " | ".join(f"{nm} {v['prose']:.0%}/{v['code']:.0%}" for nm, v in row.items() if nm != "raw")
                  + f" [{time.time()-t0:.0f}s]", flush=True)
        del s, Hp2, hin, tg


if __name__ == "__main__":
    main()
