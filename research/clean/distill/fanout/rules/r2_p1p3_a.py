"""P1-P3 r2 row a: r2_p1p3_merge.build(gate=None). See r2_p1p3_merge.py."""
from rules import r2_p1p3_merge


def vit(T, f, X=None):
    return r2_p1p3_merge.build(T, f, X, gate=None, tag="a")
