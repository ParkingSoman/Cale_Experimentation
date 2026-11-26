"""P6 diagnostic, ResNet-18: stream-only floor cut + mean-only fold (each reader's bias gets the mean of the cut channels' contribution on X). THE TEST ROW.
See _p6diag_r18.py. Input: X (generic, no photos). Photos: scoring only."""
from rules import _p6diag_r18


def r18(A, f, X=None):
    return _p6diag_r18.build(A, f, X, "mean")
