"""INPUT round 3: the CLIP stream-only ruler (P6 per-site PCA 'ps-pca', or P6 reader basis 'rG') on a chosen basis input.

Object (picked by): the stream basis of P6's builds (picked by the board: the input moves the stream basis, round 2).
Only the input X changes; builds are P6's files, imported read-only (rules/r2_p6_persite.py, rules/r2_p6_core.py).
Input spec (no photos in the calculation):
  --input A.pt[+B.pt...]   concat of generic inputs (pooled set)
  --blend w --base P.pt    X = base + w * (input - base)  (pixel blend; w > 1 extrapolates the generator's move)
  --sub N                  first N images only
  --chroma s               shrink the colour part: x = mean over channels + s (x - mean), renormalized per channel
  --aug flip               add the horizontally flipped copy of every image
  --photos K               LABELLED READING CHECK: basis from K held photos ids[1000:2000][::4][:K] (scoring uses
                           held ids[:1000] or test, disjoint); rows carry input 'heldK-CHECK', never board.json.
Scoring: --split test|held, --stride 1 (full 1,000) or 5 (quick 200). Per-photo agree saved to
out/r3_input_<tag>_agree.npy for paired reads. Rows -> fanout/scratch.jsonl, who INPUT.
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import board  # noqa: E402
from rules import r2_p6_persite as ps  # noqa: E402
from rules import r2_p6_core as core  # noqa: E402


@torch.no_grad()
def score_fwd(W, fwd, split, n=1000, stride=5, bs=100):
    """Copy of r2_p6_run.score_fwd with a split argument and per-photo agree returned."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "clipkd"))
    import eval_fast
    import vitgen
    ref = eval_fast.teacher_ref(split)
    sel = np.arange(min(n, len(ref["ids"])))[::stride]
    ref = dict(ref, ids=ref["ids"][sel], labels=ref["labels"][sel], logits=ref["logits"][sel])
    n = len(sel)
    X = eval_fast.photos(ref["ids"][:n])
    Wd = vitgen.to(W, board.DEV)
    Wd["skip"] = [(a.to(board.DEV), b.to(board.DEV)) for a, b in W["skip"]]
    cls = ref["cls"].to(board.DEV)
    lg = []
    for i in range(0, n, bs):
        e = fwd(Wd, X[i:i + bs].to(board.DEV).float()).float()
        lg.append((ref["scale"] * F.normalize(e, dim=-1) @ cls).cpu())
    lg = torch.cat(lg)
    zt = ref["logits"][:n]
    k = board.kl_rows(zt, lg)
    m = (lg.argmax(1) == zt.argmax(1)).numpy()
    return dict(kl=float(k.mean()), top1=float((lg.argmax(1).numpy() == ref["labels"][:n]).mean()),
                agree=float(m.mean()), n=n), m


def load_input(a):
    if a.photos:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "clipkd"))
        import eval_fast
        ids = eval_fast.teacher_ref("held")["ids"][1000:2000][::4][:a.photos]
        return eval_fast.photos(ids).float(), f"held{a.photos}-CHECK"
    parts = [torch.load(p, weights_only=True).float() for p in a.input.split("+")]
    X = torch.cat(parts)
    name = "+".join(os.path.basename(p)[:-3] for p in a.input.split("+"))
    if a.blend is not None:
        B = torch.load(a.base, weights_only=True).float()
        X = B + a.blend * (X - B)
        name = f"blend{a.blend:g}({name},{os.path.basename(a.base)[:-3]})"
    if a.sub:
        X = X[:a.sub]
        name += f"[:{a.sub}]"
    if a.chroma is not None:
        mu = X.mean(1, keepdim=True)
        X = mu + a.chroma * (X - mu)
        X = (X - X.mean((2, 3), keepdim=True)) / X.std((2, 3), keepdim=True)
        name += f"+chroma{a.chroma:g}"
    if a.aug == "flip":
        X = torch.cat([X, X.flip(3)])
        name += "+flip"
    return X, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="")
    ap.add_argument("--blend", type=float, default=None)
    ap.add_argument("--base", default="distill/fanout/cache/input_pink_n256_s1.pt")
    ap.add_argument("--sub", type=int, default=0)
    ap.add_argument("--aug", default="")
    ap.add_argument("--chroma", type=float, default=None, help="x = mean_c + s (x - mean_c), then per-image per-channel mean 0 std 1 (s 0 = grey)")
    ap.add_argument("--photos", type=int, default=0)
    ap.add_argument("--f", default="0.85,0.7")
    ap.add_argument("--kind", default="pca", help="pca | rand (ps per-site) | rG (P6 reader basis, rho divisor)")
    ap.add_argument("--split", default="held", choices=("test", "held"))
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    T = board.clip_teacher()
    X, inp = load_input(a)
    for f in [float(v) for v in a.f.split(",")]:
        t0 = time.time()
        if a.kind == "rG":
            W = core.build(T, f, X, "rG", "rho")
            res, _, m = board.clip_score(W, 1000, a.split, stride=a.stride, with_match=True)
            rule = "r2_p6_rG_rho"
        else:
            W = ps.build(T, f, X, a.kind)
            res, m = score_fwd(W, ps.forward, a.split, 1000, a.stride)
            rule = f"r2_p6_ps-{a.kind}"
        tag = f"{rule}_f{f:g}_{a.split}{res['n']}_{inp}".replace("/", "_").replace(":", "")[:180]
        np.save(os.path.join(HERE, "out", f"r3_input_{tag}_agree.npy"), m)
        row = dict(net="clip", rule=rule, f=f, split=a.split, input=inp, **res,
                   calc_seconds=round(time.time() - t0, 1),
                   photo_use=f"{a.split} photos scoring only" + ("; HELD PHOTOS IN THE BASIS (labelled reading check)" if a.photos else ""),
                   when=datetime.datetime.now().isoformat(timespec="seconds"), who="INPUT", quick=a.stride > 1,
                   note="r3 INPUT stream ruler " + a.note)
        print(f"ROW {rule} f={f} [{inp}] {a.split}{res['n']} KL {res['kl']:.3f} agree {100 * res['agree']:.1f}% "
              f"top1 {100 * res['top1']:.1f}% ({row['calc_seconds']} s)", flush=True)
        with open(os.path.join(HERE, "scratch.jsonl"), "a") as fh:
            fh.write(json.dumps(row) + "\n")
        del W
        if board.DEV == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
