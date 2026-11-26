"""P6 round 2 stream-only rule: basis rG (reader-weighted, no back-map), divisor rho. See r2_p6_core.py. Input X only; photos scoring only."""
from rules import r2_p6_core


def vit(T, f, X=None):
    return r2_p6_core.build(T, f, X, "rG", "rho")
