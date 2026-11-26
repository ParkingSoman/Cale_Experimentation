"""P4 write, CLIP heads-only row: only the heads set is cut (floor ranking) and folded. Stream and MLP at full width,
MLP writes stay the teacher's. See p4write.py."""
from rules import p4write


def vit(T, f, X=None):
    return p4write.build_vit(T, f, X, sets=("heads",), tag="heads")[0]
