"""TREE r3 wrapper: tilt alpha=0.1, partner=oneperm, renorm (LEAD 17:01: one kept unit per cut partner). PHOTO USE: none."""
from rules import r3_tree_lib as L


def vit(T, f, X=None):
    return L.build(T, f, X, alpha=0.1, part="oneperm", tag="onep10", renorm=True)
