"""Floor split: cut only the stream set (see _split.py)."""
from rules import _split


def vit(T, f):
    return _split.vit(T, f, "stream")


def r18(A, f):
    return _split.r18(A, f, "stream")
