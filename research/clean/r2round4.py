"""HOW round 4: a loss-aware LOCAL fit (E8, E9). No full-model training run.

One gradient pass on the ORIGINAL model gives, for each calibration token t, g_t = d(loss)/d(layer-L MLP output).
Then only the small layer is fitted, on stored inputs, to minimize
    mean_t (g_t . (Y_t - Yhat_t))^2 / A0  +  beta * mean_t ||Y_t - Yhat_t||^2 / B0
(first-order change of the loss per token, plus a small plain term; A0, B0 = values at the start).
beta large = the old local MSE fit. The fit takes seconds to minutes on the Mac.

  python3 research/clean/r2round4.py --layer 28 [--grad_text prose|code|mix --mix 0.5]
Writes results/clean-r2round4-L{L}.json (rows appended).
"""
import argparse
import json
import os
import time

import numpy as np
import mlx.core as mx
import mlx.optimizers as optim

from r2lib import Setup, ROOT, hidden
from r2round2 import ce_blocks
from r1 import SmallMLP, capture
from tree_lib import SEQ, token_blocks

HERE = os.path.dirname(os.path.abspath(__file__))


def text_blocks(tok, path, n_seq, skip=0):
    text = open(path).read()
    ids = tok.encode(text[skip:skip + 3_000_000])
    need = n_seq * (SEQ + 1)
    assert len(ids) >= need, (len(ids), need)
    return mx.array(np.array(ids[:need], dtype=np.int32).reshape(n_seq, SEQ + 1))


def grads_and_inputs(S, blocks, batch=4):
    """Per-token loss gradient at layer L's MLP output and the MLP input, original model. First position dropped."""
    L = S.L
    mlp = S.model.model.layers[L].mlp

    class Hook:
        def __init__(self, inner):
            self.inner, self.delta, self.x = inner, None, None

        def __call__(self, x):
            self.x = x
            return self.inner(x) + self.delta

    hk = Hook(mlp)
    S.model.model.layers[L].mlp = hk
    Gs, Xs = [], []
    for i in range(0, blocks.shape[0], batch):
        b = blocks[i:i + batch]
        x, y = b[:, :-1], b[:, 1:]

        def loss(delta):
            hk.delta = delta
            lg = S.model(x)
            lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            return -mx.take_along_axis(lg, y[..., None], axis=-1).sum()

        g = mx.grad(loss)(mx.zeros((x.shape[0], x.shape[1], 576)))
        mx.eval(g, hk.x)
        Gs.append(np.array(g, dtype=np.float32)[:, 1:, :].reshape(-1, 576))
        Xs.append(np.array(hk.x, dtype=np.float32)[:, 1:, :].reshape(-1, 576))
    S.model.model.layers[L].mlp = mlp
    return np.concatenate(Xs), np.concatenate(Gs)


def fit(X, Y, Gr, init, beta, steps=3000, lr=1e-3, batch=4096, seed=0):
    rng = np.random.default_rng(seed)
    P = {k: mx.array(v.astype(np.float32)) for k, v in zip("GUD", init)}
    Xm, Ym, Gm = mx.array(X), mx.array(Y), mx.array(Gr)

    def parts(P, x, y, g):
        h = x @ P["G"].T
        r = y - ((h * mx.sigmoid(h)) * (x @ P["U"].T)) @ P["D"].T
        return ((g * r).sum(-1) ** 2).mean(), (r * r).sum(-1).mean()

    idx0 = mx.array(rng.integers(0, X.shape[0], 16384))
    a0, b0 = [v.item() for v in parts(P, Xm[idx0], Ym[idx0], Gm[idx0])]

    def loss(P, x, y, g):
        a, b = parts(P, x, y, g)
        return a / a0 + beta * b / b0

    vg = mx.value_and_grad(loss)
    opt = optim.Adam(learning_rate=optim.cosine_decay(lr, steps, lr * 0.05))
    for _ in range(steps):
        idx = mx.array(rng.integers(0, X.shape[0], batch))
        _, gr = vg(P, Xm[idx], Ym[idx], Gm[idx])
        opt.update(P, gr)
        mx.eval(P, opt.state)
    return tuple(np.array(P[k], dtype=np.float64) for k in "GUD")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--grad_text", default="prose", choices=["prose", "code", "mix"])
    ap.add_argument("--mix", type=float, default=0.5, help="share of code tokens when grad_text=mix")
    ap.add_argument("--inits", default="cut,trainedGU")
    ap.add_argument("--betas", default="0.1,1000")
    ap.add_argument("--eval", default="prose", help="comma list: prose,code")
    args = ap.parse_args()
    S = Setup(args.layer, args.K, "global")
    t0 = time.time()
    n_cal = 64
    prose = token_blocks(S.tok, "train", n_cal)
    code = text_blocks(S.tok, os.path.join(HERE, "data", "code_train.txt"), n_cal)
    if args.grad_text == "prose":
        cal = prose
    elif args.grad_text == "code":
        cal = code
    else:
        nc = int(round(args.mix * n_cal))
        cal = mx.concatenate([code[:nc], prose[:n_cal - nc]]) if 0 < nc < n_cal else (code if nc else prose)
    X, Gr = grads_and_inputs(S, cal)
    Xd = X.astype(np.float64)
    Y = (hidden(Xd, S.G, S.U) @ S.D.T).astype(np.float32)
    print(f"grads ({args.grad_text}) {X.shape} [{time.time() - t0:.0f}s]", flush=True)

    evals = {"prose": token_blocks(S.tok, "test", 100),
             "code": text_blocks(S.tok, os.path.join(HERE, "data", "code_test.txt"), 60)}
    orig = S.model.model.layers[S.L].mlp
    base = {k: float(ce_blocks(S.model, v).mean()) for k, v in evals.items() if k in args.eval.split(",")}

    def ce_of(P):
        S.model.model.layers[S.L].mlp = SmallMLP(P)
        r = {k: float(ce_blocks(S.model, evals[k]).mean()) for k in base}
        S.model.model.layers[S.L].mlp = orig
        return r

    imp = np.abs(S.Hc).mean(0) * np.linalg.norm(S.D, axis=0)
    keep = np.sort(np.argsort(-imp)[:args.K])
    inits = {"cut": (S.G[keep], S.U[keep], S.D[:, keep]), "trainedGU": S.P}
    path = os.path.join(ROOT, "results", f"clean-r2round4-L{S.L}.json")
    out = json.load(open(path)) if os.path.exists(path) else {"rows": []}
    ref = {"trained_global": ce_of(S.P)}
    print(f"original {base} | trained global layer {ref['trained_global']}", flush=True)
    for iname in args.inits.split(","):
        for beta in [float(b) for b in args.betas.split(",")]:
            t1 = time.time()
            P = fit(X, Y, Gr, inits[iname], beta, steps=args.steps)
            ce = ce_of(P)
            row = {"layer": S.L, "K": args.K, "grad_text": args.grad_text, "mix": args.mix, "init": iname,
                   "beta": beta, "steps": args.steps, "secs": time.time() - t1, "ce": ce, "base": base, "ref": ref}
            out["rows"].append(row)
            json.dump(out, open(path, "w"), indent=1)
            print(f"init {iname:9s} beta {beta:7.1f} grads {args.grad_text}{'' if args.grad_text != 'mix' else args.mix} "
                  + " ".join(f"{k} +{ce[k] - base[k]:.4f}" for k in ce)
                  + f" | trained " + " ".join(f"{k} +{ref['trained_global'][k] - base[k]:.4f}" for k in ce)
                  + f" [{time.time() - t1:.0f}s fit+eval]", flush=True)


if __name__ == "__main__":
    main()
