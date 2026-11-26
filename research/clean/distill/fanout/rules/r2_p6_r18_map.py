"""P6 round 2 ResNet stream-only: read maps folded into readers, mode map. See r2_p6_r18core.py. Input X; photos scoring only."""
from rules import r2_p6_r18core


def r18(A, f, X=None):
    return r2_p6_r18core.build(A, f, X, "map")
