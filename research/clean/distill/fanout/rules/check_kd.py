"""CHECK ONLY, NOT A CALCULATION: the CLIP-KD student (trained, ViT-T/16) through the board's own forward, scored
with the teacher's class list. Target row for the ViT-T shape (f = 0.25). CALC 2.1: 1.719 on 5,000, 1.801 on 1,000."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "clipkd"))


def vit(T, f):
    assert f == 0.25
    import vitgen
    from score_clip import MODELS, build
    m = build(*MODELS["kd"])
    return vitgen.from_open_clip(m.visual)
