"""P5 r2 baseline: CLIP whole cut, stream and heads by the floor's activity on X, MLP by WEIGHT only (a = 0)."""
from rules import r2_p5_lib as L


def vit(T, f, X=None):
    return L.build(T, f, X, 1.0, ("stream", "heads", "mlp"), {"mlp": 0.0})
