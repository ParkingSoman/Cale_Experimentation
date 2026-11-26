"""Floor split: cut only the mlp set (see _split.py)."""
from rules import _split


def vit(T, f):
    return _split.vit(T, f, "mlp")
