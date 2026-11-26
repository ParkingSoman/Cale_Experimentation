"""TREE r3 wrapper: tilt alpha=0.05, partner=rand (see r3_tree_lib). PHOTO USE: none."""
from rules import r3_tree_lib as L


def vit(T, f, X=None):
    return L.build(T, f, X, alpha=0.05, part="rand", tag="rand05")
