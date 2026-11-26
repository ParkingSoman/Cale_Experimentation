"""Weight-ranking split: CLIP MLP neurons only, magcut score (see _magsplit.py)."""
from rules import _magsplit


def vit(T, f):
    return _magsplit.vit(T, f, "mlp")
