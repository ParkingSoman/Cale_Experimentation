"""TREE r3 wrapper: tilt alpha=0.1, partner=one, renorm (LEAD 17:01: one kept unit per cut partner). PHOTO USE: none."""
from rules import r3_tree_lib as L


def vit(T, f, X=None):
    return L.build(T, f, X, alpha=0.1, part="one", tag="one10", renorm=True)
