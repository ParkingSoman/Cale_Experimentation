"""INPUT round 3: JOINT's r2_joint_floor_sfon_on (floor order + closed-loop stream fold + inner fold), with the
ORDER fixed on the shared pink file and only the FOLD statistics taken from the board's --input X.
Object (picked by): the fold statistics of the ResNet stream fold (picked by the lead's round-3 task: does the input
matter for the stream fold, as it does for CLIP's stream basis?). Order on pink so a row reads the fold input alone.
Photo use: none. Weights + generic input only. BatchNorm running stats never read as data.
"""
from rules import r2_joint_core as J
from rules import _p6diag_r18 as P6

_orig_keep = J.keep_sets


def r18(A, f, X=None):
    pink = P6._input(None)

    def keep_pink(A_, f_, X_, order):
        return _orig_keep(A_, f_, pink, order)
    J.keep_sets = keep_pink
    try:
        return J.build_r18(A, f, X, order="floor", fold="on", sfold="on")
    finally:
        J.keep_sets = _orig_keep
