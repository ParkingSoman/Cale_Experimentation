"""N7 diagnostic 2: are the sequential builds under-confident (regression dilution)?

For B, T, seq, lens: logits on held (2,000) and test (1,500).
  spread   = std of centred logits of X / that of Base (median over photos). Below 1 = flatter, under-confident
  cos      = cosine of (X - Base) centred logits with Base's centred logits (median). Below 0 = shrunk toward the mean
  temp     = one logit scale fitted on held by grid; test KL with it
Saved: results/lens/diag2.json
"""
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gap"))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import build as LB  # noqa: E402
from train_ref import load_mlp_state  # noqa: E402

torch.set_grad_enabled(False)


def logits(m, cache):
    out = []
    for x, _ in cache.batches(50):
        out.append(C.tail(m, x).cpu())
    return torch.cat(out)


def main():
    lab = C.net.Photos("read").lab
    base = C.load("base")
    calib = C.calibrate(base, (x for x, _ in C.Cache("cal").batches(25)))
    nets = {}
    mB = copy.deepcopy(base)
    C.build(mB, calib, 0.5, "value")
    nets["B"] = mB
    ck = torch.load(os.path.join(C.OUT, "T.pt"), weights_only=False)
    mT = copy.deepcopy(base)
    C.build(mT, calib, ck["frac"], ck["how"])
    load_mlp_state(mT, ck["state"])
    nets["T"] = mT
    fitc = C.Cache("train")
    xs, _ = fitc.get(LB.fit_rows(fitc, 4, lab))
    xs = xs.half()
    specs = [a.split(":") for a in sys.argv[1:]] or [["seq", "seq", "1e-3", "0", "0"], ["lens", "lens", "1e-4", "0", "0"]]
    for tag, mode, rl, otl, otw in specs:
        nets[tag], _ = LB.build_lens(base, calib, xs, mode, 0.5, float(rl), 1e-6, 30, False, None,
                                     otl == "1", otw == "1")
        torch.save(dict(state=[nets[tag].features[s].state_dict() for s in (5, 6, 7)],
                        head=nets[tag].classifier.state_dict()), os.path.join(LB.OUT, f"{tag}_saved.pt"))
    del xs
    held, te = C.Cache("held"), C.Cache("test")
    zh0, zt0 = torch.from_numpy(held.z).float(), torch.from_numpy(te.z).float()
    res = {}
    grid = np.arange(0.5, 2.51, 0.05)
    for name, mX in nets.items():
        zh, zt = logits(mX, held), logits(mX, te)
        c0 = zt0 - zt0.mean(1, keepdim=True)
        cx = zt - zt.mean(1, keepdim=True)
        spread = float((cx.std(1) / c0.std(1)).median())
        cos = float(F.cosine_similarity(cx - c0, c0, dim=1).median())
        kh = [float(C.kl_rows(zh0, zh * t).mean()) for t in grid]
        t = float(grid[int(np.argmin(kh))])
        res[name] = dict(test_kl=float(C.kl_rows(zt0, zt).mean()), spread=spread, cos=cos, temp=t,
                         test_kl_temp=float(C.kl_rows(zt0, zt * t).mean()))
        print(name, res[name], flush=True)
    json.dump(res, open(os.path.join(LB.OUT, "diag2_" + "_".join(t[0] for t in specs) + ".json" if sys.argv[1:] else "diag2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
