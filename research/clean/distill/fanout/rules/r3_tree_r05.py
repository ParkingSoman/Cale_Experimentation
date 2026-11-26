"""TREE r3 wrapper: tilt alpha=.05 toward the top R1 partner (r3_tree_r1lib). PHOTO USE: none."""
from rules import r3_tree_lib as L, r3_tree_r1lib as R


def vit(T, f, X=None):
    return L.build(T, f, X, alpha=0.05, part="r1", r1=R.make(T), tag="r05")
