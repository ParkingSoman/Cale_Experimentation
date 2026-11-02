"""Baseline: cut (keep the K biggest-output neurons) + least-squares output refit. No fit beyond LS."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import mlp_io, cut_ls, SwiGLULayer  # noqa: E402

METHOD = {"name": "cut_ls", "family": "baseline-cut", "minutes": 0.5}


def build(model, tok, L, K, cal):
    X, _ = mlp_io(model, cal[:64], L)
    G, U, D, _ = cut_ls(model, L, K, X)
    return SwiGLULayer(G, U, D)
