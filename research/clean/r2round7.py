"""HOW round 7: how much lookahead does the short fit need? (blue sky #7: cost for large models)

Layer L's small MLP is fitted so that the model still matches the original W layers later:
  W = "end": KL on next-word probabilities through every later layer (round 5)
  W = w:     relative MSE of the residual stream after layer L+w (block-wise reconstruction with lookahead)
Same start (cut), same wall time. Cost per step grows with the window.

  python3 research/clean/r2round7.py --layer 15 --windows 1,3,7,end --minutes 2
"""
import argparse
import json
import os
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm.models.base import create_attention_mask

from r2lib import Setup, ROOT
from r2round3 import joint_out
from r2round5 import cache_h
from r2round2 import ce_blocks
from r1 import SmallMLP
from tree_lib import token_blocks


def run_from(S, h, mlp, upto):
    """Residual after layer `upto` (inclusive), starting from h_L with the given MLP at L. upto=None: logits."""
    m = S.model.model
    b = m.layers[S.L]
    out = h + mlp(b.post_attention_layernorm(h))
    mask = create_attention_mask(out, None)
    last = len(m.layers) - 1 if upto is None else upto
    for l in range(S.L + 1, last + 1):
        out = m.layers[l](out, mask)
    if upto is None:
        return m.embed_tokens.as_linear(m.norm(out))
    return out


def fit_window(S, H, init, window, minutes, lr=3e-4, batch=4, seed=0):
    rng = np.random.default_rng(seed)
    student, teacher = SmallMLP(init), S.model.model.layers[S.L].mlp
    last = len(S.model.model.layers) - 1
    # "endmse": the full tail, scored by residual MSE at the last layer (control: window vs objective type).
    upto = None if window == "end" else (last if window == "endmse" else min(S.L + int(window), last))
    S.model.freeze()
    student.unfreeze()

    def loss_fn(student, h, tgt):
        y = run_from(S, h, student, upto)
        if upto is None:
            y = y - mx.logsumexp(y, axis=-1, keepdims=True)
            return (mx.exp(tgt) * (tgt - y)).sum(-1)[:, 1:].mean()
        r = (y - tgt)[:, 1:]
        return (r * r).sum() / ((tgt[:, 1:] - tgt[:, 1:].mean(-1, keepdims=True)) ** 2).sum()

    vg = nn.value_and_grad(student, loss_fn)
    opt = optim.Adam(learning_rate=lr)
    t0, steps = time.time(), 0
    while time.time() - t0 < minutes * 60:
        h = mx.array(H[rng.integers(0, H.shape[0], batch)])
        tgt = run_from(S, h, teacher, upto)
        if upto is None:
            tgt = tgt - mx.logsumexp(tgt, axis=-1, keepdims=True)
        tgt = mx.stop_gradient(tgt)
        _, g = vg(student, h, tgt)
        opt.update(student, g)
        mx.eval(student.parameters(), opt.state)
        steps += 1
    S.model.unfreeze()
    return tuple(np.array(w, dtype=np.float64) for w in
                 (student.gate_proj.weight, student.up_proj.weight, student.down_proj.weight)), steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--windows", default="1,3,7,end")
    ap.add_argument("--minutes", type=float, default=2)
    args = ap.parse_args()
    S = Setup(args.layer, args.K, "global")
    t0 = time.time()
    H = cache_h(S, token_blocks(S.tok, "train", 64))
    held = token_blocks(S.tok, "test", 100)
    orig = S.model.model.layers[S.L].mlp
    base = float(ce_blocks(S.model, held).mean())

    def ce_of(P):
        S.model.model.layers[S.L].mlp = SmallMLP(P)
        r = float(ce_blocks(S.model, held).mean())
        S.model.model.layers[S.L].mlp = orig
        return r

    imp = np.abs(S.Hc).mean(0) * np.linalg.norm(S.D, axis=0)
    keep = np.sort(np.argsort(-imp)[:args.K])
    G0, U0 = S.G[keep], S.U[keep]
    P0 = (G0, U0, joint_out(S.Xc, G0, U0, S.Hc @ S.D.T))
    res = {"layer": S.L, "K": args.K, "minutes": args.minutes, "base": base, "cut": ce_of(P0),
           "trained_global": ce_of(S.P), "rows": {}}
    print(f"L{S.L}: cut +{res['cut'] - base:.4f}, trained global (20 min, full model) "
          f"+{res['trained_global'] - base:.4f}", flush=True)
    for w in args.windows.split(","):
        P, steps = fit_window(S, H, P0, w, args.minutes)
        ce = ce_of(P)
        res["rows"][w] = {"ce": ce, "steps": steps}
        print(f"window {w:>4s}: +{ce - base:.4f} after {args.minutes} min ({steps} steps) [{time.time() - t0:.0f}s]",
              flush=True)
        json.dump(res, open(os.path.join(ROOT, "results", f"clean-r2round7-L{S.L}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
