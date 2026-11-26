"""P5 r2: CLIP MLP-only cut, score = weight^(1-a) x activity^a, a = .50 (see r2_p5_lib)."""
from rules import r2_p5_lib as L


def vit(T, f, X=None):
    return L.build(T, f, X, .50, ("mlp",))
