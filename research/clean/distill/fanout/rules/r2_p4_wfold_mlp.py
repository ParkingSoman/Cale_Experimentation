"""P4 round 2: CLIP MLP-only fold on the WEIGHT order (magcut scores, P5's a = 0 endpoint) instead of the floor order.
Object (picked by): pr_w of every block; kept MLP units = magcut weight ranking. Fold = r2_p4_lib (slice anchor).
PHOTO USE: none."""
import torch

from rules import magcut, r2_p4_lib as L
import vitgen


def keep_w(T, f, which=("mlp",)):
    W = magcut.vit_scores(T)
    d, h, m = vitgen.sizes(f, T)
    D, H, M = vitgen.sizes(1.0, T)
    return dict(stream=torch.arange(D) if "stream" not in which else magcut.top(W["stream"], d),
                heads=[magcut.top(s, h if "heads" in which else H) for s in W["head"]],
                mlp=[magcut.top(s, m if "mlp" in which else M) for s in W["mlp"]])


def vit(T, f, X=None):
    return L.build_vit2(T, f, X, sets=("mlp",), keep=keep_w(T, f), tag="wfold_mlp")
