"""P4 round 2: heads-only, same-form MEAN null. See r2_p4_lib.py."""
from rules import r2_p4_lib as L


def vit(T, f, X=None):
    return L.build_vit2(T, f, X, sets=("heads",), null="mean", tag="r2_p4_meannull_heads")
