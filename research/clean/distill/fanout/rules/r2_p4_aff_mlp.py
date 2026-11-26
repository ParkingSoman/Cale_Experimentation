"""P4 round 2: MLP-only fold, AFFINE (bias column). See r2_p4_lib.py."""
from rules import r2_p4_lib as L


def vit(T, f, X=None):
    return L.build_vit2(T, f, X, sets=("mlp",), affine=True, tag="r2_p4_aff_mlp")
