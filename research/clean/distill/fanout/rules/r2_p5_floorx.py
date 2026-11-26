"""P5 r2 baseline: CLIP whole cut, floor form (activity on X for every set, a = 1) through r2_p5_lib."""
from rules import r2_p5_lib as L


def vit(T, f, X=None):
    return L.build(T, f, X, 1.0, ("stream", "heads", "mlp"))
