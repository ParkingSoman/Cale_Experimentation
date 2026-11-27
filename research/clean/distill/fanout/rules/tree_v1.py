"""Tree v1 (distill/PAGE.md v1), whole build, loo=None. Photo use: none (weights + generic input)."""
from rules import _tree


def vit(T, f, X=None):
    return _tree.vit(T, f, X, loo=None)


def r18(A, f, X=None):
    return _tree.r18(A, f, X, loo=None)
