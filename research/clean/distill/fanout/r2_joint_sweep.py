"""JOINT round 2: run many r2_joint variants in ONE torch job (one lock wait). Scoring = board.py's own functions.
Quick: 200 test photos (stride 5), rows to scratch.jsonl. Full (--full): 1,000 photos, rows to board.json.
Usage: python -I distill/fanout/r2_joint_sweep.py --rules r2_joint_floor_off_off,... --f 0.95,0.85 [--full --note ..]
Photo use: test photos, scoring only. The calculation uses weights + the generic input X only."""
import argparse
import datetime
import importlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import board  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rules", required=True)
ap.add_argument("--net", default="r18", choices=("r18", "clip"))
ap.add_argument("--f", default="0.95,0.85")
ap.add_argument("--full", action="store_true")
ap.add_argument("--note", default="")
ap.add_argument("--input", default=os.path.join(HERE, "cache", "input_pink_n256_s1.pt"))
a = ap.parse_args()
X = torch.load(a.input, weights_only=True).float()
inp = os.path.basename(a.input)[:-3]
A = board.r18_A() if a.net == "r18" else board.clip_teacher()
if a.net == "clip":
    import vitgen
stride = 1 if a.full else 5
for rn in a.rules.split(","):
    fn = getattr(importlib.import_module(f"rules.{rn}"), "r18" if a.net == "r18" else "vit")
    for f in [float(v) for v in a.f.split(",")]:
        t0 = time.time()
        S = fn(A, f, X=X)
        tc = time.time() - t0
        res, k = board.r18_score(S, 1000, "test", stride=stride) if a.net == "r18" else board.clip_score(S, 1000, "test", stride=stride)
        row = dict(net=a.net, rule=rn, f=f, split="test", input=inp, **res,
                   params=sum(p.numel() for p in S.parameters()) if a.net == "r18" else vitgen.n_params(S), shape=dict(widths=[64, 128, 256, 512]) if a.net == "r18" else {},
                   calc_seconds=round(tc, 1), total_seconds=round(time.time() - t0, 1),
                   photo_use="test photos, scoring only", when=datetime.datetime.now().isoformat(timespec="seconds"),
                   note=a.note, who="JOINT", quick=not a.full)
        if a.full:
            np.save(os.path.join(board.OUTD, f"{a.net}_{rn}_f{f:g}_test{res['n']}_{inp}.npy"), k)
            board.add_row(row)
        else:
            with open(os.path.join(HERE, "scratch.jsonl"), "a") as fh:
                fh.write(json.dumps(row) + "\n")
        print(f"ROW {rn} f={f:g} KL {res['kl']:.3f} agree {res['agree']:.3f} top1 {res['top1']:.3f} ({row['total_seconds']} s)", flush=True)
        del S
