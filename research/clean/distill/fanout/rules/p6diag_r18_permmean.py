"""P6 diagnostic, ResNet-18, NULL for meanfold: the same fold computed with the cut channels' activations permuted among the cut channels (derangement).
See _p6diag_r18.py. Input: X (generic, no photos). Photos: scoring only."""
from rules import _p6diag_r18


def r18(A, f, X=None):
    return _p6diag_r18.build(A, f, X, "perm")
