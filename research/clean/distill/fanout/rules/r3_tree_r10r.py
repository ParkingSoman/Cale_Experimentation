"""TREE r3 wrapper: tilt alpha=0.1, partner=r1, renorm (tilted pre-activation keeps the kept unit mean and sd). PHOTO USE: none."""
from rules import r3_tree_lib as L, r3_tree_r1lib as R  # noqa: F401


def vit(T, f, X=None):
    return L.build(T, f, X, alpha=0.1, part="r1", tag="r10r", renorm=True, r1=R.make(T))
