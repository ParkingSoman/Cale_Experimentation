"""P4 write, CLIP MLP-only row: only the MLP set is cut (floor ranking) and folded. Stream and heads at full width,
head writes stay the teacher's. See p4write.py."""
from rules import p4write


def vit(T, f, X=None):
    return p4write.build_vit(T, f, X, sets=("mlp",), tag="mlp")[0]
