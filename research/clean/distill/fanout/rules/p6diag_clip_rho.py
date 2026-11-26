"""P6 diagnostic, CLIP: stream-only floor cut, centred stream, rms norms with divisor D_l = 768 * rho_l (per-site kept energy share on X). THE TEST ROW.
See _p6diag_clip.py. Input: X (generic, no photos). Photos: scoring only."""
from rules import _p6diag_clip


def vit(T, f, X=None):
    return _p6diag_clip.build(T, f, X, "rho")
