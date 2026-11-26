"""Floor split: cut only the inner set (see _split.py)."""
from rules import _split


def r18(A, f):
    return _split.r18(A, f, "inner")
