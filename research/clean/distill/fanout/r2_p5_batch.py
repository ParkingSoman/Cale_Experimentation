"""P5 r2 batch runner: ONE torch process, many quick rows (the shared lock waits ~6 min per call).
Usage: board-like specs  NET:RULE[:ATTR]:F1,F2  ...  (ATTR = a function name in the rule module, default vit/r18).
Rows go to fanout/scratch.jsonl (quick, 200 photos) or board.json with --full. Scoring = board.py's own functions."""
import argparse
import datetime
import importlib
import inspect
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import board  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("specs", nargs="+")
ap.add_argument("--full", action="store_true")
ap.add_argument("--note", default="")
ap.add_argument("--split", default="test", choices=("test", "held"))
ap.add_argument("--input", default="distill/fanout/cache/input_pink_n256_s1.pt")
a = ap.parse_args()
X = torch.load(a.input, weights_only=True).float()
inp = os.path.basename(a.input)[:-3]
big = {}
for spec in a.specs:
    p = spec.split(":")
    net, rule, fs = p[0], p[1], p[-1]
    attr = p[2] if len(p) == 4 else ("vit" if net == "clip" else "r18")
    if net not in big:
        big[net] = board.clip_teacher() if net == "clip" else board.r18_A()
    fn = getattr(importlib.import_module(f"rules.{rule}"), attr)
    tx = "X" in inspect.signature(fn).parameters
    for f in [float(v) for v in fs.split(",")]:
        t0 = time.time()
        try:
            small = fn(big[net], f, X=X) if tx else fn(big[net], f)
        except Exception as e:  # noqa: BLE001
            print(f"ERR {spec} {f}: {e!r}", flush=True)
            continue
        tc = time.time() - t0
        if net == "clip":
            res, k = board.clip_score(small, 1000, a.split, stride=1 if a.full else 5)
        else:
            res, k = board.r18_score(small, 1000, a.split, stride=1 if a.full else 5)
        if net == "clip":
            import vitgen
            params, shp = vitgen.n_params(small), vitgen.shape(small)
            shp = dict(d=shp["d"], heads=sorted(set(shp["heads"])), mlp=sorted(set(shp["mlp"])))
        else:
            params = sum(q.numel() for q in small.parameters())
            shp = dict(widths=list(getattr(small, "w", (64, 128, 256, 512))))
        name = rule + ("" if attr in ("vit", "r18") else "." + attr)
        row = dict(net=net, rule=name, f=f, split=a.split, input=inp if tx else "none", **res, params=params, shape=shp, calc_seconds=round(tc, 1),
                   total_seconds=round(time.time() - t0, 1), photo_use=f"{a.split} photos, scoring only",
                   when=datetime.datetime.now().isoformat(timespec="seconds"), note=a.note, who="P5", quick=not a.full)
        if a.full:
            import numpy as np
            np.save(os.path.join(board.OUTD, f"{net}_{name}_f{f:g}_{a.split}{res['n']}_{inp}.npy"), k)
            board.add_row(row)
        else:
            with open(os.path.join(HERE, "scratch.jsonl"), "a") as fh:
                fh.write(json.dumps(row) + "\n")
        print(f"ROW {net} {name} f={f} kl={res['kl']:.3f} top1={res['top1']:.3f} agree={res['agree']:.3f} calc={tc:.0f}s", flush=True)
        del small
        if board.DEV == "mps":
            torch.mps.empty_cache()
