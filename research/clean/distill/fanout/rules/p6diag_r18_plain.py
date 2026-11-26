"""P6 diagnostic, ResNet-18, BASELINE on X: the floor's stream-only cut, no fold, stream order from X. Same as split_stream but the order comes from the input X.
See _p6diag_r18.py. Input: X (generic, no photos). Photos: scoring only."""
from rules import _p6diag_r18


def r18(A, f, X=None):
    return _p6diag_r18.build(A, f, X, "plain")
