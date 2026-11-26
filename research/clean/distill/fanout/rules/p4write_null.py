"""P4 write NULL, whole network: shuffled pairing (cut-unit teacher firings from the next image, same positions;
kept-to-kept pairing true). In expectation the position-wise mean-only fold. See p4write.py."""
from rules import p4write


def vit(T, f, X=None):
    return p4write.build_vit(T, f, X, sets=("stream", "heads", "mlp"), null=True, tag="null_full")[0]


def r18(A, f, X=None):
    return p4write.build_r18(A, f, X, sets=("stream", "inner"), null=True, tag="null_full")[0]
