"""Liquid track harness (plan ~/.claude/plans/okay-right-now-i-crystalline-platypus.md). Dev text only.

Shared pieces for shrinking attention heads, head size and stream width, scored in one currency:
added next-word loss (dCE) on dev prose + dev code, per share of weights removed.
Reuses the coarsening track's code by import only (never edits it).
"""
import json, os, sys, time
import numpy as np
import mlx.core as mx, mlx.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CLEAN))
sys.path[:0] = [CLEAN, os.path.join(CLEAN, "coarse"), os.path.join(CLEAN, "coarse", "methods"), os.path.join(ROOT, "research", "tree")]
from tree_lib import load_model, token_blocks, SEQ  # noqa: E402
from r1 import block_ce  # noqa: E402
from rcf_v1 import _capture, _cv_ridge, _ridge, SinkGuard, Lin, silu  # noqa: E402
from contract import orig_weights, SwiGLULayer  # noqa: E402

D_MODEL, N_HEADS, N_KV, HEAD_DIM, D_FF = 576, 9, 3, 64, 1536
LAYER_PARAMS = 2 * D_MODEL * N_HEADS * HEAD_DIM + 2 * D_MODEL * N_KV * HEAD_DIM + 3 * D_MODEL * D_FF  # 3,538,944
HEAD_PARAMS = 2 * HEAD_DIM * D_MODEL      # one query head: q rows + o columns = 73,728
NEURON_PARAMS = 3 * D_MODEL               # one MLP neuron = 1,728
DEV_LAYERS = (12, 16, 22)                 # never 6, 9, 21, 24 (keeper layers)


def code_blocks(tok, n, start_chars):
    txt = open(os.path.join(CLEAN, "data", "code_train.txt")).read()[start_chars:start_chars + 1200000]
    ids = np.array(tok.encode(txt), dtype=np.int32)
    n = min(n, len(ids) // (SEQ + 1))
    return mx.array(ids[:n * (SEQ + 1)].reshape(n, SEQ + 1))


class Setup:
    """Model + calibration (prose train + code chars 400k+, as rcf_v1) + dev (prose test 0-39, code chars 0-400k)."""

    def __init__(self, n_cal=64, n_dev=40):
        self.model, self.tok = load_model()
        prose_cal = token_blocks(self.tok, "train", 256)[:n_cal]
        self.cal = mx.concatenate([prose_cal, code_blocks(self.tok, n_cal, 400000)], 0)
        self.dev = {"prose": token_blocks(self.tok, "test", n_dev),
                    "code": code_blocks(self.tok, n_dev, 0)}
        self.base = {k: block_ce(self.model, v) for k, v in self.dev.items()}

    def score(self, edits):
        """edits = [(owner, attr, new_module)]. Apply, score dev, restore. Returns {text: per-block dCE array}."""
        old = [(o, a, _get(o, a)) for o, a, _ in edits]
        for o, a, m in edits:
            _set(o, a, m)
        try:
            return {k: block_ce(self.model, v) - self.base[k] for k, v in self.dev.items()}
        finally:
            for o, a, m in reversed(old):
                _set(o, a, m)


def _get(o, a):
    return o[a] if isinstance(o, list) else getattr(o, a)


def _set(o, a, m):
    if isinstance(o, list):
        o[a] = m
    else:
        setattr(o, a, m)


def boot(d, n=2000, seed=0):
    r = np.random.default_rng(seed)
    m = np.array([d[r.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return [float(d.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def save(name, rows):
    path = os.path.join(ROOT, "results", f"liquid-{name}.json")
    old = json.load(open(path)) if os.path.exists(path) else {"rows": []}
    old["rows"] += rows
    json.dump(old, open(path, "w"), indent=1)
    return path


class Timer:
    def __init__(self):
        self.t = time.time()

    def __call__(self):
        s = time.time() - self.t; self.t = time.time(); return round(s, 1)
