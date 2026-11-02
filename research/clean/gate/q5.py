"""Q5: loss-weighted rows for the RCF solve (CHILDREN-LOG Q5). Row weight w_t = |d(next-word loss)/d(MLP L output)_t|^2
from one backward pass of the original model on the calibration blocks (mean 1). Closed form, single layer.

  python3 research/clean/gate/q5.py --layers 16,22 --Ks 768
"""
import argparse, json, os, sys, time
import numpy as np
import mlx.core as mx, mlx.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from a0 import grams, cv_ridge_g, LAMS, FORBID, ROOT  # noqa: E402
from q1 import setup, Installer  # noqa: E402
from tree_lib import load_model  # noqa: E402
from r1 import block_ce  # noqa: E402
from g6 import boot  # noqa: E402
from g12 import dev_sets, mixed_cal  # noqa: E402


class AddDelta(nn.Module):
    def __init__(self, inner):
        super().__init__(); self.inner, self.delta = inner, None
    def __call__(self, x):
        return self.inner(x) + self.delta


def loss_weights(model, blocks, L, batch=4):
    lay = model.model.layers[L]; w = AddDelta(lay.mlp); lay.mlp = w; out = []
    for i in range(0, blocks.shape[0], batch):
        b = blocks[i:i + batch]; inp, tgt = b[:, :-1], b[:, 1:]

        def f(delta):
            w.delta = delta; lg = model(inp); lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            return -mx.take_along_axis(lg, tgt[..., None], axis=-1).sum()
        g = mx.grad(f)(mx.zeros((inp.shape[0], inp.shape[1], lay.mlp.inner.down_proj.weight.shape[0])))
        out.append(np.array((g * g).sum(-1)))
    lay.mlp = w.inner
    W = np.concatenate(out, 0)[:, 1:].reshape(-1)     # rows as capture: block-major, position 0 dropped
    return (W / W.mean()).astype(np.float32)


def wgrams(A, Y, w):
    s = np.sqrt(w)[:, None]; return grams(A * s, Y * s)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="16,22"); ap.add_argument("--Ks", default="768"); ap.add_argument("--ncal", type=int, default=128)
    a = ap.parse_args(); layers = [int(x) for x in a.layers.split(",")]; assert not FORBID & set(layers)
    model, tok = load_model(); cal = mixed_cal(tok, a.ncal); dev, devc = dev_sets(tok)
    texts = (("prose", dev), ("code", devc)); base = {t: block_ce(model, b) for t, b in texts}
    path = os.path.join(ROOT, "results", "clean-q5.json"); out = json.load(open(path)) if os.path.exists(path) else {}; t0 = time.time()
    for L in layers:
        s = setup(model, L, cal); ins = Installer(model, L); R = np.concatenate([s["Hp"], s["Z"]], 1); P = s["Hp"].shape[1]
        wt = loss_weights(model, cal, L); assert len(wt) == len(s["Y"])
        print(f"L{L} weights: max/mean {wt.max():.0f}, top-1% rows hold {np.sort(wt)[-len(wt)//100:].sum()/wt.sum():.0%} [{time.time()-t0:.0f}s]", flush=True)
        for K in [int(k) for k in a.Ks.split(",")]:
            keep = np.sort(np.argsort(-s["en"])[:K]); Hk = s["H"][:, keep]; A = np.concatenate([Hk, R], 1)
            Ws = {}
            Ws["plain"], _ = cv_ridge_g(grams(A, s["Y"]), LAMS)
            Ws["w"], _ = cv_ridge_g(wgrams(A, s["Y"], wt), LAMS)
            Ws["sw"], _ = cv_ridge_g(wgrams(A, s["Y"], np.sqrt(wt)), LAMS)
            Wc, _ = cv_ridge_g(grams(Hk, s["Y"]), (1e-4,)); Wcw, _ = cv_ridge_g(wgrams(Hk, s["Y"], np.sqrt(wt)), (1e-4,)); del A
            res = {}
            ins.install(s, keep, Wc.T, None, None, "fold"); res["cut"] = {t: block_ce(model, b) for t, b in texts}; ins.restore()
            ins.install(s, keep, Wcw.T, None, None, "fold"); res["cut_sw"] = {t: block_ce(model, b) for t, b in texts}; ins.restore()
            for nm, Wa in Ws.items():
                for place in ("fold", "oracle"):
                    ins.install(s, keep, Wa[:K].T, Wa[K:K + P], Wa[K + P:], place); res[f"{place}_{nm}"] = {t: block_ce(model, b) for t, b in texts}; ins.restore()
            row = {}
            for t, _ in texts:
                c = float((res["cut"][t] - base[t]).mean()); row[t] = {"cut": c}
                for arm in res:
                    if arm != "cut":
                        v = float((res[arm][t] - base[t]).mean()); row[t][arm] = {"dce": v, "share": 1 - v / c}
                row[t]["fold_w_minus_plain"] = boot(res["fold_w"][t] - res["fold_plain"][t]); row[t]["fold_sw_minus_plain"] = boot(res["fold_sw"][t] - res["fold_plain"][t])
            out[f"L{L}_K{K}"] = row; json.dump(out, open(path, "w"), indent=1)
            for t, _ in texts:
                print(f"L{L} K{K} {t}: " + " ".join(f"{k} {v['share']:.0%}" for k, v in row[t].items() if isinstance(v, dict) and "share" in v)
                      + f" | w-plain {np.round(row[t]['fold_w_minus_plain'], 4)} sw-plain {np.round(row[t]['fold_sw_minus_plain'], 4)}", flush=True)


if __name__ == "__main__":
    main()
