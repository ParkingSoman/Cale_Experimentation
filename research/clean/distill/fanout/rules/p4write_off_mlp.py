"""P4 control: CLIP MLP-only plain slice (fold OFF) through p4write's code path, on the given X. Must reproduce
split_mlp (same draw up to float16). Implementation check of the index plumbing."""
from rules import p4write


def vit(T, f, X=None):
    return p4write.build_vit(T, f, X, sets=("mlp",), fold=False, tag="off_mlp")[0]
