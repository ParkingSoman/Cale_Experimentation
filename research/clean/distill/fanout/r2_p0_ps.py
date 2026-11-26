"""P0 round 2: P6's per-site PCA stream-only build (rules/r2_p6_persite.py, read-only import) on a chosen basis input.

Object (picked by): the per-site stream bases of P6's ps-pca build (picked by the lead, NOTES 15:32: the sharpest input
test, since photo bases leave chance at 50% and pink bases do not). Only the input X changes; build and score are P6's.
--input FILE.pt : a generic input (no photos in the calculation).
--photos K      : LABELLED READING CHECK: the basis from K held photos (same ids as P6's --photo, first K); rows carry
                  input 'heldK-CHECK' and never go to board.json.
Rows: quick (200 test photos), appended to fanout/scratch.jsonl with who P0.
"""
import argparse
import datetime
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import board  # noqa: E402
import r2_p6_run as p6  # noqa: E402
from rules import r2_p6_persite as ps  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="")
    ap.add_argument("--photos", type=int, default=0)
    ap.add_argument("--f", default="0.85,0.7,0.5")
    ap.add_argument("--kind", default="pca")
    a = ap.parse_args()
    T = board.clip_teacher()
    if a.photos:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "clipkd"))
        import eval_fast
        ids = eval_fast.teacher_ref("held")["ids"][1000:2000][::4][:a.photos]
        X = eval_fast.photos(ids).float()
        inp = f"held{a.photos}-CHECK"
    else:
        X = torch.load(a.input, weights_only=True).float()
        inp = os.path.basename(a.input)[:-3]
    for f in [float(v) for v in a.f.split(",")]:
        t0 = time.time()
        W = ps.build(T, f, X, a.kind)
        res, _ = p6.score_fwd(W, ps.forward, 1000, 5)
        row = dict(net="clip", rule=f"r2_p6_ps-{a.kind}", f=f, split="test", input=inp, **res,
                   calc_seconds=round(time.time() - t0, 1),
                   photo_use="test photos scoring only" + ("; HELD PHOTOS IN THE BASIS (labelled reading check)" if a.photos else ""),
                   when=datetime.datetime.now().isoformat(timespec="seconds"), who="P0", quick=True, note="r2 P0 input test of P6 ps build")
        print(f"ROW ps-{a.kind} f={f} [{inp}] KL {res['kl']:.3f} agree {100 * res['agree']:.1f}% top1 {100 * res['top1']:.1f}%", flush=True)
        with open(os.path.join(HERE, "scratch.jsonl"), "a") as fh:
            fh.write(json.dumps(row) + "\n")
        del W
        if board.DEV == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
