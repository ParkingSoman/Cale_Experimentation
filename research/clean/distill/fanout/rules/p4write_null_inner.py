"""P4 write NULL, ResNet inner-only: shuffled pairing on the cut inner channels. See p4write.py."""
from rules import p4write


def r18(A, f, X=None):
    return p4write.build_r18(A, f, X, sets=("inner",), null=True, tag="null_inner")[0]
