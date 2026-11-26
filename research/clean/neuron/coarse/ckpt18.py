"""N11 Q5: capture B k128 seed 0's block-0 neurons at each saved checkpoint (watch the coarse neurons form).
Saved: results/coarse/r18/ckpt_read_B128aug_s0.npz with pos0_<step> and arg0_<step> (19,000 held + test photos)."""
import glob
import os
import re

import numpy as np
import torch

import r18
from capture18 import capture

torch.set_grad_enabled(False)
A = r18.load_a()
m = r18.make_b(A, 128, 0)
caches = [r18.Cache("held"), r18.Cache("test")]
out = {}
for f in sorted(glob.glob(os.path.join(r18.OUT, "ckpt", "B128aug_s0_*.pt")), key=lambda p: int(re.findall(r"_(\d+)\.pt", p)[0])):
    step = int(re.findall(r"_(\d+)\.pt", f)[0])
    r18.load_state(m, torch.load(f, weights_only=False)["state"])
    parts = [capture(m.eval(), c) for c in caches]
    out[f"pos0_{step}"] = np.concatenate([p["pos0"] for p in parts])
    out[f"arg0_{step}"] = np.concatenate([p["arg0"] for p in parts])
    print("checkpoint", step, flush=True)
np.savez(os.path.join(r18.OUT, "ckpt_read_B128aug_s0.npz"), **out)
