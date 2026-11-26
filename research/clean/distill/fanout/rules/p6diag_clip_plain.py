"""P6 diagnostic, CLIP, BASELINE on X: the floor's stream-only cut (LayerNorm over the kept dims), stream order from X. Same as split_stream but the order comes from the input X.
See _p6diag_clip.py. Input: X (generic, no photos). Photos: scoring only."""
from rules import _p6diag_clip


def vit(T, f, X=None):
    return _p6diag_clip.build(T, f, X, "plain")
