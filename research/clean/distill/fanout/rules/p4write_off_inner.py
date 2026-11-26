"""P4 control: ResNet inner-only plain slice (fold OFF, cut inner = zero conv2 column) through p4write's code path,
on the given X. The floor for p4write_inner on the SAME input (split_inner used a different pink draw)."""
from rules import p4write


def r18(A, f, X=None):
    return p4write.build_r18(A, f, X, sets=("inner",), fold=False, tag="off_inner")[0]
