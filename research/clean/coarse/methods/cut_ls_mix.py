"""Baseline: cut_ls with the SAME mixed calibration rcf_v1 uses (cal prose blocks + the same number of code blocks from
code_train.txt chars 400k+). Needed for a fair code comparison with rcf_v1. No fit beyond LS."""
import os, sys
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import mlp_io, cut_ls, SwiGLULayer, ROOT  # noqa: E402

METHOD = {"name": "cut_ls_mix", "family": "baseline-cut", "minutes": 1.0}
N_CAL = 64


def build(model, tok, L, K, cal):
    txt = open(os.path.join(ROOT, "research", "clean", "data", "code_train.txt")).read()[400000:]
    ids = np.array(tok.encode(txt[:1200000]), dtype=np.int32)[: N_CAL * 513]
    blocks = mx.concatenate([cal[:N_CAL], mx.array(ids.reshape(N_CAL, 513))], 0)
    X, _ = mlp_io(model, blocks, L)
    G, U, D, _ = cut_ls(model, L, K, X)
    return SwiGLULayer(G, U, D)
