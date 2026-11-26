"""P1-P3 r2 row f: r2_p1p3_merge.build(tree="r1w", gate="or"). See r2_p1p3_merge.py."""
from rules import r2_p1p3_merge


def vit(T, f, X=None):
    return r2_p1p3_merge.build(T, f, X, tree="r1w", gate="or", tag="f")
