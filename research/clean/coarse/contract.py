"""Coarsening track: the contract every method follows (plan ~/.claude/plans/ok-id-like-you-unified-blum.md).

A method is ONE python file with:
  METHOD = {"name": str, "family": str, "minutes": float}   # family = the idea; minutes = declared compute budget
  def build(model, tok, L, K, cal) -> CoarseLayer

  model  the intact original model (mlx_lm, float32). The method may read any weight and run it on `cal`.
  L, K   layer index and number of coarse units.
  cal    calibration token blocks (wikitext-2 TRAIN). Public text only.

Not allowed: trained parents for any layer (research/clean/r1_weights, load_w), anything in ~/Cale-holdout.

CoarseLayer is an mlx nn.Module that replaces model.model.layers[L].mlp:
  __call__(x)  -> (..., d)  the layer output
  units(x)     -> (..., K)  one scalar activation per coarse unit (used by the cover test)
  n_params()   -> int       stored numbers (cost test compares with a cut of equal size)
"""
import importlib.util
import os
import sys

import numpy as np
import mlx.core as mx
import mlx.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "research", "tree"))
from tree_lib import SEQ, hidden, ls_down  # noqa: E402

FORBIDDEN = ("r1_weights", "load_w", "Cale-holdout", "r1family", "r1cover.json")


def load_method(path):
    src = open(path).read()
    bad = [w for w in FORBIDDEN if w in src]
    if bad:
        raise ValueError(f"method {path} uses forbidden inputs: {bad}")
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert {"name", "family", "minutes"} <= set(mod.METHOD), "METHOD needs name, family, minutes"
    return mod


class _Grab(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner, self.x = inner, []

    def __call__(self, x):
        self.x.append(np.array(x).reshape(-1, x.shape[-1]))
        return self.inner(x)


def mlp_io(model, blocks, L, batch=4):
    """MLP inputs X and outputs Y at layer L on `blocks`, first position of each block dropped (attention sink)."""
    layer = model.model.layers[L]
    g = _Grab(layer.mlp)
    layer.mlp = g
    for i in range(0, blocks.shape[0], batch):
        mx.eval(model(blocks[i:i + batch, :-1]))
    layer.mlp = g.inner
    X = np.concatenate(g.x, 0)
    keep = np.arange(X.shape[0]) % SEQ != 0
    X = X[keep].astype(np.float64)
    return X, None


def resid_in(model, blocks, L, batch=4):
    """Residual stream h entering layer L's MLP sub-block (input of post_attention_layernorm), all positions."""
    layer = model.model.layers[L]
    g = _Grab(layer.post_attention_layernorm)
    layer.post_attention_layernorm = g
    for i in range(0, blocks.shape[0], batch):
        mx.eval(model(blocks[i:i + batch, :-1]))
    layer.post_attention_layernorm = g.inner
    return np.concatenate(g.x, 0).reshape(blocks.shape[0], SEQ, -1)


def orig_weights(model, L):
    m = model.model.layers[L].mlp
    return (np.array(m.gate_proj.weight, dtype=np.float64), np.array(m.up_proj.weight, dtype=np.float64),
            np.array(m.down_proj.weight, dtype=np.float64))


class SwiGLULayer(nn.Module):
    """K-unit SwiGLU layer (the same form as the original)."""

    def __init__(self, G, U, D):
        super().__init__()
        self.G, self.U, self.D = (mx.array(np.asarray(a, dtype=np.float32)) for a in (G, U, D))

    def units(self, x):
        return nn.silu(x @ self.G.T) * (x @ self.U.T)

    def __call__(self, x):
        return self.units(x) @ self.D.T

    def n_params(self):
        return int(self.G.size + self.U.size + self.D.size)


def cut_ls(model, L, K, X):
    """Keep the K neurons with the largest output on X, least-squares refit of their outputs."""
    G, U, D = orig_weights(model, L)
    h = hidden(X, G, U)
    score = np.linalg.norm(h, axis=0) * np.linalg.norm(D, axis=0)
    idx = np.sort(np.argsort(-score)[:K])
    Y = h @ D.T
    return G[idx], U[idx], ls_down(X, G[idx], U[idx], Y).astype(np.float64), idx
