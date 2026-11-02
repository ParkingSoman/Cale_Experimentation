"""Baseline: cut_ls start, then a 3-minute fit of the K-unit layer only (base frozen) so the residual stream
3 layers later matches the original (R2.7: a 3-layer window is best at fixed time)."""
import os, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm.models.base import create_attention_mask
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import mlp_io, resid_in, cut_ls, SwiGLULayer  # noqa: E402

METHOD = {"name": "shortfit3", "family": "baseline-shortfit", "minutes": 3.5}
WINDOW, MINUTES, LR, BATCH = 3, 3.0, 3e-4, 4


def build(model, tok, L, K, cal):
    X, _ = mlp_io(model, cal[:64], L)
    G, U, D, _ = cut_ls(model, L, K, X)
    student = SwiGLULayer(G, U, D)
    H = resid_in(model, cal[:256], L)
    m = model.model
    block, teacher = m.layers[L], m.layers[L].mlp
    upto = min(L + WINDOW, len(m.layers) - 1)

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
    vg = nn.value_and_grad(student, loss_fn)
    opt = optim.Adam(learning_rate=LR)
    rng = np.random.default_rng(0)
    t0 = time.time()
    while time.time() - t0 < MINUTES * 60:
        h = mx.array(H[rng.integers(0, H.shape[0], BATCH)])
        tgt = mx.stop_gradient(run(h, teacher))
        _, g = vg(student, h, tgt)
        opt.update(student, g)
        mx.eval(student.parameters(), opt.state)
    model.unfreeze()
    return student
