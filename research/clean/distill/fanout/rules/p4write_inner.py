"""P4 write, ResNet inner-only row: only the inner set is cut (floor ranking) and folded into conv2. Stream at full
width. See p4write.py."""
from rules import p4write


def r18(A, f, X=None):
    return p4write.build_r18(A, f, X, sets=("inner",), tag="inner")[0]
