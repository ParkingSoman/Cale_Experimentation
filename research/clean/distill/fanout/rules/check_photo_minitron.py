"""CHECK ONLY, NOT A CALCULATION (it uses photo-based scores). Rebuilds the logged plain cut of distill/LOG.md
(Minitron scores on 2,000 train photos, saved in distill_whole/W<pct>P_s0_scores.pt) so board.py can be checked
against the logged held KL before training (95% 1.113, 50% 8.413, 5% 7.543)."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
D = os.path.join(CLEAN, "neuron", "results", "coarse", "r18", "distill_whole")


def r18(A, f):
    import wide18
    sc = torch.load(os.path.join(D, f"W{round(100 * f)}P_s0_scores.pt"), weights_only=False)["scores"]
    S, _ = wide18.cut_from_a(A, f, sc)
    return S.eval()
