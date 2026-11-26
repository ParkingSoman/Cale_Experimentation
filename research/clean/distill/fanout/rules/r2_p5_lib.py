"""P5 round 2 (fast loop): rankings that mix weight size and activity on the generic input.
Part changed: P5 Rank only. No weight is changed (plain cut through vitgen.cut).
Object (picked by): CLIP MLP neurons / heads / stream dims (every unit of the set; picked by the score below).
Derived from: BOARD F1 + round-1 review (weight ranking beat the activity ranking on CLIP MLP, .205 vs .323/.473).
Score family (same form, one exponent a): s = W^(1-a) * ACT^a, W = magcut weight score, ACT = noisecut activity.
a = 0 is the weight ranking, a = 1 is the floor's activity ranking (the two endpoint rows are the nulls of the form).
Photo use: NONE. Weights + the generic input X (board --input) or noisecut's default pink.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import vitgen  # noqa: E402
from rules import magcut, noisecut  # noqa: E402

_C = {}


def scores(T, X=None):
    key = "x" if X is not None else "pink"
    if key not in _C:
        _C[key] = (magcut.vit_scores(T), noisecut.vit_scores(T, X=X))
    return _C[key]


def mixs(w, act, a):
    return (w.clamp_min(1e-12).log() * (1 - a) + act.clamp_min(1e-12).log() * a)


def build(T, f, X, a, which=("mlp",), a_set=None):
    """which: sets that are cut. a_set: {set: a} overrides."""
    W, A = scores(T, X)
    a_set = a_set or {}
    d, h, m = vitgen.sizes(f, T)
    D, H, M = vitgen.sizes(1.0, T)
    d = d if "stream" in which else D
    h = h if "heads" in which else H
    m = m if "mlp" in which else M
    st = mixs(W["stream"], A["stream"], a_set.get("stream", a))
    hd = [mixs(w, x, a_set.get("heads", a)) for w, x in zip(W["head"], A["head"])]
    ml = [mixs(w, x, a_set.get("mlp", a)) for w, x in zip(W["mlp"], A["mlp"])]
    return vitgen.cut(T, magcut.top(st, d), [magcut.top(s, h) for s in hd], [magcut.top(s, m) for s in ml])
