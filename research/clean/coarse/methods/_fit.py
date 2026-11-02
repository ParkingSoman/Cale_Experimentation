"""Shared short fit: train a CoarseLayer's free weights (base frozen) so the residual 3 layers later matches the
original. Same recipe as shortfit3 (window 3, Adam 3e-4, batch 4, 256 cal blocks)."""
import os, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm.models.base import create_attention_mask
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import resid_in  # noqa: E402


def fit_window(model, L, student, cal, minutes=3.0, freeze=(), window=3, lr=3e-4, batch=4, seed=0):
    H = resid_in(model, cal[:256], L)
    m = model.model
    block, teacher = m.layers[L], m.layers[L].mlp
    upto = min(L + window, len(m.layers) - 1)

    def run(h, mlp):
        out = h + mlp(block.post_attention_layernorm(h))
        mask = create_attention_mask(out, None)
        for l in range(L + 1, upto + 1):
            out = m.layers[l](out, mask)
        return out

    def loss_fn(st, h, tgt):
        r = (run(h, st) - tgt)[:, 1:]
        c = tgt[:, 1:] - tgt[:, 1:].mean(-1, keepdims=True)
        return (r * r).sum() / (c * c).sum()

    model.freeze()
    student.unfreeze()
    if freeze:
        student.freeze(keys=list(freeze))
    vg = nn.value_and_grad(student, loss_fn)
    opt = optim.Adam(learning_rate=lr)
    rng = np.random.default_rng(seed)
    t0, steps = time.time(), 0
    while time.time() - t0 < minutes * 60:
        h = mx.array(H[rng.integers(0, H.shape[0], batch)])
        tgt = mx.stop_gradient(run(h, teacher))
        _, g = vg(student, h, tgt)
        opt.update(student, g)
        mx.eval(student.parameters(), opt.state)
        steps += 1
    model.unfreeze()
    student.unfreeze()
    return student, steps
