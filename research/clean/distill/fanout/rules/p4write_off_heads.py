"""P4 control: CLIP heads-only plain slice (fold OFF) through p4write's code path, on the given X."""
from rules import p4write


def vit(T, f, X=None):
    return p4write.build_vit(T, f, X, sets=("heads",), fold=False, tag="off_heads")[0]
