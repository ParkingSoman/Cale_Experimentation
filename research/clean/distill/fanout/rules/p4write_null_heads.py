"""P4 write NULL, CLIP heads-only: shuffled pairing on the cut head outputs. See p4write.py."""
from rules import p4write


def vit(T, f, X=None):
    return p4write.build_vit(T, f, X, sets=("heads",), null=True, tag="null_heads")[0]
