"""P6 diagnostic, CLIP, NULL for rho: same as rho but one global rho (mean of the 26 rho_l) at every site.
See _p6diag_clip.py. Input: X (generic, no photos). Photos: scoring only."""
from rules import _p6diag_clip


def vit(T, f, X=None):
    return _p6diag_clip.build(T, f, X, "grho")
