"""Weight-ranking split rows (the P5 fail line): magcut scores (weights only), ONE CLIP set cut, the others at full
width, through the same vitgen.cut as rules/_split.py. Photo use: none. No generic input either (weights only)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import vitgen  # noqa: E402
from rules import magcut  # noqa: E402


def vit(T, f, which):
    sc = magcut.vit_scores(T)
    d, h, m = vitgen.sizes(f, T)
    D, H, M = vitgen.sizes(1.0, T)
    d, h, m = (d if which == "stream" else D), (h if which == "heads" else H), (m if which == "mlp" else M)
    return vitgen.cut(T, magcut.top(sc["stream"], d), [magcut.top(s, h) for s in sc["head"]], [magcut.top(s, m) for s in sc["mlp"]])
