"""P5 r2: CLIP MLP-only cut, weight^.75 x activity^.25 (a picked on 200 quick test photos; see r2_p5_w25)."""
from rules import r2_p5_lib as L


def vit(T, f, X=None):
    return L.build(T, f, X, .25, ("mlp",))
