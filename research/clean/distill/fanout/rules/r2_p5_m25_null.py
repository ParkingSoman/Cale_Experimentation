"""P5 r2 NULL: CLIP MLP-only, same form as r2_p5_m25 with the MLP activity permuted inside each block."""
from rules import r2_p5_lib as L


def vit(T, f, X=None):
    return L.build(T, f, X, .25, ("mlp",), null=True)
