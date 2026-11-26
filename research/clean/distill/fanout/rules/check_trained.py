"""CHECK ONLY, NOT A CALCULATION: the trained targets of distill/LOG.md section 8, scored on the board's photos
so the bar sits on the same photos as every rule. 95% = W95P_s0, 50% = W50P_s0_x2, 5% = W5R_s0_x2.
Re-run after Shonu's wave 3 retrain with the new best files."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
D = os.path.join(CLEAN, "neuron", "results", "coarse", "r18", "distill_whole")
FILES = {0.95: "W95P_s0.pt", 0.5: "W50P_s0_x2.pt", 0.05: "W5R_s0_x2.pt"}


def r18(A, f):
    import wide18
    S = wide18.make_student(f)
    sd = torch.load(os.path.join(D, FILES[f]), map_location="cpu", weights_only=False)
    sd = sd["state"]
    S.load_state_dict(sd)
    return S.eval()
