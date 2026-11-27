"""L2 gate law, ResNet-18, gate=perm. Round-3 best (r3_input_r18sf: floor order on pink, stream + inner folds,
fold statistics from the board's --input X) + the gate switch. See rules/l2_gate_r18core.py. Photo use: none."""
from rules import l2_gate_r18core as J
from rules import _p6diag_r18 as P6

_orig_keep = J.keep_sets


def r18(A, f, X=None):
    pink = P6._input(None)

    def keep_pink(A_, f_, X_, order):
        return _orig_keep(A_, f_, pink, order)
    J.keep_sets = keep_pink
    try:
        return J.build_r18(A, f, X, order="floor", fold="on", sfold="on", gate="perm")
    finally:
        J.keep_sets = _orig_keep
