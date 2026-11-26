"""N7: run the N6 error split (gap/additive.py) on a saved lens-folder build whose blocks are fitted on Base's inputs.
Usage: python split.py base_otw_saved   -> results/lens/additive_<tag>.json"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gap"))
import common as C  # noqa: E402

tag = sys.argv[1]
OUT = os.path.join(HERE, "..", "results", "lens")
ck = torch.load(os.path.join(OUT, f"{tag}.pt"), weights_only=False)
ck.update(frac=0.5, how="value")
torch.save(ck, os.path.join(OUT, f"{tag}_split.pt"))
C.OUT = OUT
import additive  # noqa: E402
sys.argv = ["additive.py", "--run", f"{tag}_split", "--only", "T"]
additive.main()
os.remove(os.path.join(OUT, f"{tag}_split.pt"))
