"""P4 round 2: same-form MEAN null of r2_p4_wfold_mlp (weight order). PHOTO USE: none."""
from rules import r2_p4_lib as L
from rules.r2_p4_wfold_mlp import keep_w


def vit(T, f, X=None):
    return L.build_vit2(T, f, X, sets=("mlp",), keep=keep_w(T, f), null="mean", tag="wmeannull_mlp")
