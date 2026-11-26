"""P4 write NULL, CLIP MLP-only: shuffled pairing on the cut MLP units. See p4write.py."""
from rules import p4write


def vit(T, f, X=None):
    return p4write.build_vit(T, f, X, sets=("mlp",), null=True, tag="null_mlp")[0]
