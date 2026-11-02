"""HOW round 5: short-tail distillation (blue sky #5) and task-specific parents (multiple-parents hypothesis).

Store the residual stream h_L (after layer L's attention, before its MLP) for calibration text, once.
Then fit only the small MLP so that the model's next-word distribution matches the original one, running ONLY
the layers after L (for L = 28: two layers plus the head), not the whole model:
    out = h_L + mlp(norm(h_L)) -> layers L+1..end -> logits ;  loss = KL(original || small)
The calibration text sets the objective: prose, code, or a mix with share `mix` of code.

  python3 research/clean/r2round5.py --layer 28 --texts prose,code,mix0.5 --minutes 2
Writes results/clean-r2round5-L{L}.json (rows appended).
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
from r2round2 import ce_blocks
from r2round4 import text_blocks
from r1 import SmallMLP, load_w
from tree_lib import token_blocks

HERE = os.path.dirname(os.path.abspath(__file__))


class LayerHook(nn.Module):
    """Wraps a transformer block and stores h (after attention, before the MLP)."""

    def __init__(self, inner):
        super().__init__()
        self.inner, self.h = inner, []
        self.use_sliding = getattr(inner, "use_sliding", False)

    def __call__(self, x, mask=None, cache=None):
        b = self.inner
        h = x + b.self_attn(b.input_layernorm(x), mask, cache)
        self.h.append(np.array(h.astype(mx.float32)))
        return h + b.mlp(b.post_attention_layernorm(h))


def cache_h(S, blocks, batch=4):
    layers = S.model.model.layers
    hk = LayerHook(layers[S.L])
    layers[S.L] = hk
    for i in range(0, blocks.shape[0], batch):
        mx.eval(S.model(blocks[i:i + batch, :-1]))
    layers[S.L] = hk.inner
    return np.concatenate(hk.h, 0)                        # n_blocks x T x 576


def tail(S, h, mlp):
    """Logits from h_L through layer L's MLP (given) and every later layer."""
    m = S.model.model
    b = m.layers[S.L]
    out = h + mlp(b.post_attention_layernorm(h))
    mask = create_attention_mask(out, None)
    for l in range(S.L + 1, len(m.layers)):
        out = m.layers[l](out, mask)
    return m.embed_tokens.as_linear(m.norm(out))


def fit_tail(S, H, init, minutes, lr=3e-4, batch=4, seed=0, frozen=()):
    """frozen: names of student parts kept fixed, e.g. ("gate_proj",) or ("gate_proj", "up_proj")."""
    rng = np.random.default_rng(seed)
    student = SmallMLP(init)
    teacher = S.model.model.layers[S.L].mlp
    S.model.freeze()
    student.unfreeze()
    for name in frozen:
        getattr(student, name).freeze()

    def loss_fn(student, h, tl):
        lg = tail(S, h, student)
        lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        return (mx.exp(tl) * (tl - lg)).sum(-1)[:, 1:].mean()

    vg = nn.value_and_grad(student, loss_fn)
    opt = optim.Adam(learning_rate=lr)
    t0, steps = time.time(), 0
    while time.time() - t0 < minutes * 60:
        h = mx.array(H[rng.integers(0, H.shape[0], batch)])
        tl = tail(S, h, teacher)
        tl = mx.stop_gradient(tl - mx.logsumexp(tl, axis=-1, keepdims=True))
        _, g = vg(student, h, tl)
        opt.update(student, g)
        mx.eval(student.parameters(), opt.state)
        steps += 1
    S.model.unfreeze()
    return tuple(np.array(w, dtype=np.float64) for w in
                 (student.gate_proj.weight, student.up_proj.weight, student.down_proj.weight)), steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--texts", default="prose,code,mix0.5")
    ap.add_argument("--init", default="cut", choices=["cut", "local_cut"])
    ap.add_argument("--minutes", type=float, default=2)
    ap.add_argument("--n_cal", type=int, default=64)
    args = ap.parse_args()
    S = Setup(args.layer, args.K, "global")
    t0 = time.time()
    prose = token_blocks(S.tok, "train", args.n_cal)
    code = text_blocks(S.tok, os.path.join(HERE, "data", "code_train.txt"), args.n_cal)
    Hp, Hc = cache_h(S, prose), cache_h(S, code)
    print(f"cached h: prose {Hp.shape}, code {Hc.shape} [{time.time() - t0:.0f}s]", flush=True)

    evals = {"prose": token_blocks(S.tok, "test", 100),
             "code": text_blocks(S.tok, os.path.join(HERE, "data", "code_test.txt"), 60)}
    orig = S.model.model.layers[S.L].mlp
    base = {k: float(ce_blocks(S.model, v).mean()) for k, v in evals.items()}

    def ce_of(P):
        S.model.model.layers[S.L].mlp = SmallMLP(P)
        r = {k: float(ce_blocks(S.model, v).mean()) for k, v in evals.items()}
        S.model.model.layers[S.L].mlp = orig
        return r

    if args.init == "cut":
        imp = np.abs(S.Hc).mean(0) * np.linalg.norm(S.D, axis=0)
        keep = np.sort(np.argsort(-imp)[:args.K])
        from r2round3 import joint_out
        G0, U0 = S.G[keep], S.U[keep]
        init = (G0, U0, joint_out(S.Xc, G0, U0, S.Hc @ S.D.T))
    else:
        init = load_w(S.L, args.K, "local_cut")
    ref = {"trained_global": ce_of(S.P), "init": ce_of(init)}
    print(f"original {base}\n trained global {ref['trained_global']}\n init ({args.init}) {ref['init']}", flush=True)
    path = os.path.join(ROOT, "results", f"clean-r2round5-L{S.L}.json")
    out = json.load(open(path)) if os.path.exists(path) else {"rows": []}
    for name in args.texts.split(","):
        if name == "prose":
            H = Hp
        elif name == "code":
            H = Hc
        else:
            m = float(name[3:])
            nc = int(round(m * Hp.shape[0]))
            H = np.concatenate([Hc[:nc], Hp[:Hp.shape[0] - nc]])
        P, steps = fit_tail(S, H, init, args.minutes)
        ce = ce_of(P)
        tag = f"tail_{name}_{args.init}"
        np.savez(os.path.join(HERE, "r1_weights", f"L{S.L}_K{args.K}_{tag}.npz"), G=P[0], U=P[1], D=P[2])
        row = {"layer": S.L, "K": args.K, "text": name, "init": args.init, "minutes": args.minutes, "steps": steps,
               "ce": ce, "base": base, "ref": ref}
        out["rows"].append(row)
        json.dump(out, open(path, "w"), indent=1)
        print(f"fit on {name:8s} ({steps} steps, {args.minutes} min): prose +{ce['prose'] - base['prose']:.4f}  "
              f"code +{ce['code'] - base['code']:.4f}   | trained global prose "
              f"+{ref['trained_global']['prose'] - base['prose']:.4f} code "
              f"+{ref['trained_global']['code'] - base['code']:.4f} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
