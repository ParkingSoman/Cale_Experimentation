"""Tree v1 (distill/PAGE.md v1), whole build, loo="rand". Photo use: none (weights + generic input)."""
from rules import _tree


def vit(T, f, X=None):
    return _tree.vit(T, f, X, loo="rand")


def r18(A, f, X=None):
    return _tree.r18(A, f, X, loo="rand")
