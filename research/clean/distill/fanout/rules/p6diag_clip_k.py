"""P6 diagnostic, CLIP: stream-only floor cut, centred stream (J folded into writers), rms norms with divisor D_l = k (the centring change alone).
See _p6diag_clip.py. Input: X (generic, no photos). Photos: scoring only."""
from rules import _p6diag_clip


def vit(T, f, X=None):
    return _p6diag_clip.build(T, f, X, "k")
