"""N7 diagnostic: does each block correct the stream error it receives (self-repair)?

For each network X (B, T, seq, lens) on held photos, per block g (stream before the block x_X, Base's x_B):
  e      = x_X - x_B                                  incoming stream error
  drift  = |e|^2 / |x_B - mean|^2                     relative size
  repair = -<d, e> / |e|^2  with d = f_X(x_X) - f_B(x_B)       share of e the block removes along e (X's own block)
  base_repair = the same with Base's full block on X's stream: d = f_B(x_X) - f_B(x_B)   (does Base itself repair?)
  out    = |d|^2 / |e|^2                               size of the block's change against the incoming error
Saved: results/lens/diag.json
"""
import copy
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gap"))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import build as LB  # noqa: E402
from train_ref import load_mlp_state  # noqa: E402

DEV = C.DEV
torch.set_grad_enabled(False)


def streams(m, x):
    """Stream before every block (30) for photos x (stage-3 input)."""
    out = []
    for s in (5, 7):
        if s == 7:
            x = m.features[6](x)
        for blk in m.features[s]:
            out.append(x)
            x = blk(x)
    return out


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
    for mode, rl in (("seq", 1e-3), ("lens", 1e-4)):
        nets[mode], _ = LB.build_lens(base, calib, xs, mode, 0.5, rl, 1e-6, 30, False, None)
    del xs
    held = C.Cache("held")
    rows = np.arange(0, 2000, 10)
    BB = C.blocks(base)
    acc = {k: {q: np.zeros(30) for q in ("ee", "xx", "de", "dd", "bde")} for k in nets}
    for a in range(0, len(rows), 25):
        x, _ = held.get(rows[a:a + 25])
        sB = streams(base, x)
        for name, mX in nets.items():
            sX = streams(mX, x)
            BX = C.blocks(mX)
            for g in range(30):
                xb, xx = sB[g].float(), sX[g].float()
                e = xx - xb
                fb = BB[g](xb) - xb
                d = (BX[g](xx) - xx) - fb
                db = (BB[g](xx) - xx) - fb
                A = acc[name]
                A["ee"][g] += float((e * e).sum())
                A["xx"][g] += float(((xb - xb.mean((0, 2, 3), keepdim=True)) ** 2).sum())
                A["de"][g] += float((d * e).sum())
                A["dd"][g] += float((d * d).sum())
                A["bde"][g] += float((db * e).sum())
    res = {}
    for name, A in acc.items():
        res[name] = dict(drift=(A["ee"] / A["xx"]).tolist(), repair=(-A["de"] / A["ee"]).tolist(),
                         base_repair=(-A["bde"] / A["ee"]).tolist(), out=(A["dd"] / A["ee"]).tolist())
        r = res[name]
        print(name, "drift", np.round(r["drift"], 3).tolist(), flush=True)
        print(name, "repair", np.round(r["repair"], 3).tolist(), flush=True)
        print(name, "base_repair", np.round(r["base_repair"], 3).tolist(), flush=True)
    json.dump(res, open(os.path.join(LB.OUT, "diag.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
