"""P6 round 2 stream-only rule: basis rand, divisor D. See r2_p6_core.py. Input X only; photos scoring only."""
from rules import r2_p6_core


def vit(T, f, X=None):
    return r2_p6_core.build(T, f, X, "rand", "D")
