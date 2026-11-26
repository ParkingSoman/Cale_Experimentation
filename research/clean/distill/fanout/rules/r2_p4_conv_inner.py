"""P4 round 2, ResNet inner-only: output-matched 3x3 conv write. See r2_p4_r18lib.py."""
from rules import r2_p4_r18lib as L


def r18(A, f, X=None):
    return L.build(A, f, X, form="conv", tag="r2_p4_conv_inner")
