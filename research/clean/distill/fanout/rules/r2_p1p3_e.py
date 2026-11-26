"""P1-P3 r2 row e: r2_p1p3_merge.build(tree="rand", gate="or"). See r2_p1p3_merge.py."""
from rules import r2_p1p3_merge


def vit(T, f, X=None):
    return r2_p1p3_merge.build(T, f, X, tree="rand", gate="or", tag="e")
