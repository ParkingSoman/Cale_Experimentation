"""P1-P3 r2 row p (shape-kept null): R1 owners permuted over the deleted units, OR gate. See r2_p1p3_merge.py."""
from rules import r2_p1p3_merge


def vit(T, f, X=None):
    return r2_p1p3_merge.build(T, f, X, tree="perm", gate="or", tag="p")
