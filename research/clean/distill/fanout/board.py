"""FANOUT scoreboard (plan 2026-11-26): build a WHOLE network at fraction f with a calculation rule, score its KL to
the big network on test photos, add one row to board.json.

Usage (from research/clean):
  curves/.venv/bin/python -I distill/fanout/board.py --net clip --rule magcut --f 0.95,0.5,0.25,0.05 [--n 1000]
  curves/.venv/bin/python -I distill/fanout/board.py --net r18  --rule magcut --f 0.95,0.5,0.05 [--n 1000]
  --rule identity --f 1 is the known-answer check (KL must be 0).
  --split held scores on the held photos instead (the only split any choice may be TUNED on).

Photos: scoring only. CLIP: saved teacher logits (clipkd/out/teacher_{test,held}.npz), the teacher's class list and
scale 100, photo order one per class first (eval_fast.teacher_ref), so --n 1000 = one photo per class.
ResNet-18: split() of distill/train_whole.py (test 5 per class, held 2 per class); --n 1000 = first test photo per
class. A's logits are computed once and cached in fanout/cache/.
Outputs: fanout/board.json (rows), fanout/out/<net>_<rule>_f<f>_<split><n>.npy (KL per photo, in photo order).
ONE torch job on the Mac at a time: only the runner calls this.
"""
import argparse
import datetime
import importlib
import inspect
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.dirname(HERE)
CLEAN = os.path.dirname(DIST)
sys.path.insert(0, HERE)
sys.path.insert(0, DIST)
CACHE = os.path.join(HERE, "cache")
OUTD = os.path.join(HERE, "out")
BOARD = os.path.join(HERE, "board.json")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def kl_rows(z_ref, z):
    lr, ls = F.log_softmax(z_ref.float(), 1), F.log_softmax(z.float(), 1)
    return (lr.exp() * (lr - ls)).sum(1)


# ---------------------------------------------------------------- CLIP
def clip_teacher():
    sys.path.insert(0, os.path.join(DIST, "clipkd"))
    import vitgen
    p = os.path.join(CACHE, "clip_teacher_visual.pt")
    if os.path.exists(p):
        return torch.load(p, weights_only=True)
    from score_clip import MODELS, build
    m = build(*MODELS["teacher"])
    T = vitgen.from_open_clip(m.visual)
    os.makedirs(CACHE, exist_ok=True)
    torch.save(T, p)
    del m
    return T


@torch.no_grad()
def clip_score(W, n, split, bs=100, stride=1, with_match=False):
    sys.path.insert(0, os.path.join(DIST, "clipkd"))
    import eval_fast
    import vitgen
    ref = eval_fast.teacher_ref(split)
    n = min(n, len(ref["ids"]))
    sel = np.arange(n)[::stride]
    ref = dict(ref, ids=ref["ids"][sel], labels=ref["labels"][sel], logits=ref["logits"][sel])
    n = len(sel)
    X = eval_fast.photos(ref["ids"][:n])
    Wd = vitgen.to(W, DEV)
    cls = ref["cls"].to(DEV)
    lg = []
    for i in range(0, n, bs):
        e = vitgen.forward(Wd, X[i:i + bs].to(DEV).float()).float()
        lg.append((ref["scale"] * F.normalize(e, dim=-1) @ cls).cpu())
    lg = torch.cat(lg)
    zt = ref["logits"][:n]
    k = kl_rows(zt, lg)
    lab = ref["labels"][:n]
    m = (lg.argmax(1) == zt.argmax(1)).numpy()
    return dict(kl=float(k.mean()), top1=float((lg.argmax(1).numpy() == lab).mean()),
                agree=float(m.mean()), n=n), k.numpy(), *((m,) if with_match else ())


# ---------------------------------------------------------------- ResNet-18
def r18_photos(split, n):
    sys.path.insert(0, os.path.join(CLEAN, "neuron", "coarse"))
    import r18
    from train_whole import split as split_fn
    ph = r18.net.Photos("read")
    rows = split_fn(ph.lab)[split]
    k = 5 if split == "test" else 2
    order = np.concatenate([np.arange(j, len(rows), k) for j in range(k)])  # one per class first
    rows = rows[order][:n]
    return r18, ph, rows


@torch.no_grad()
def r18_logits(m, x, bs=250):
    m = m.to(DEV).eval()
    return torch.cat([m(x[i:i + bs].to(DEV).float()).float().cpu() for i in range(0, len(x), bs)])


_X = {}


def r18_x(split, n, stride=1):
    if (split, n, stride) not in _X:
        r18, ph, rows = r18_photos(split, n)
        rows = rows[::stride]
        _X[(split, n, stride)] = (torch.cat([ph.tensor(rows[a:a + 500]).half() for a in range(0, len(rows), 500)]), ph.lab[rows])
    return _X[(split, n, stride)]


def r18_A():
    sys.path.insert(0, os.path.join(CLEAN, "neuron", "coarse"))
    import r18
    return r18.load_a().cpu().eval()


def r18_score(S, n, split, stride=1, with_match=False):
    x, lab = r18_x(split, n, stride)
    p = os.path.join(CACHE, f"r18_A_{split}{n}" + (f"s{stride}" if stride > 1 else "") + ".pt")
    if os.path.exists(p):
        zA = torch.load(p, weights_only=True)
    else:
        zA = r18_logits(r18_A(), x)
        os.makedirs(CACHE, exist_ok=True)
        torch.save(zA, p)
    z = r18_logits(S, x)
    k = kl_rows(zA, z)
    m = (z.argmax(1) == zA.argmax(1)).numpy()
    return dict(kl=float(k.mean()), top1=float((z.argmax(1).numpy() == lab).mean()),
                agree=float(m.mean()), n=len(lab)), k.numpy(), *((m,) if with_match else ())


# ---------------------------------------------------------------- main
def add_row(row):
    rows = json.load(open(BOARD)) if os.path.exists(BOARD) else []
    rows.append(row)
    json.dump(rows, open(BOARD, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", choices=("clip", "r18"), required=True)
    ap.add_argument("--rule", required=True)
    ap.add_argument("--f", default="1")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--split", default="test", choices=("test", "held"))
    ap.add_argument("--note", default="")
    ap.add_argument("--quick", action="store_true", help="200 photos (every 5th class of the 1,000-photo set); row goes to scratch.jsonl, not board.json")
    ap.add_argument("--who", default="lead")
    ap.add_argument("--input", default="", help="cached generic input tensor (.pt, normalized images) passed to the rule as X")
    a = ap.parse_args()
    rule = importlib.import_module(f"rules.{a.rule}")
    os.makedirs(OUTD, exist_ok=True)
    big = clip_teacher() if a.net == "clip" else r18_A()
    if a.net == "clip":
        import vitgen
    stride = 5 if a.quick else 1
    if a.quick:
        a.n = 1000
    X = torch.load(a.input, weights_only=True).float() if a.input else None
    fn = getattr(rule, "vit" if a.net == "clip" else "r18")
    takes_x = "X" in inspect.signature(fn).parameters
    if X is not None and not takes_x:
        raise SystemExit(f"rule {a.rule} does not take an input X")
    for f in [float(v) for v in a.f.split(",")]:
        t0 = time.time()
        small = fn(big, f, X=X) if takes_x else fn(big, f)
        t_calc = time.time() - t0
        if a.net == "clip":
            params, shp = vitgen.n_params(small), vitgen.shape(small)
            shp = dict(d=shp["d"], heads=sorted(set(shp["heads"])), mlp=sorted(set(shp["mlp"])))
            res, k, match = clip_score(small, a.n, a.split, stride=stride, with_match=True)
        else:
            params = sum(p.numel() for p in small.parameters())
            shp = dict(widths=list(getattr(small, "w", (64, 128, 256, 512))))
            res, k, match = r18_score(small, a.n, a.split, stride=stride, with_match=True)
        inp = os.path.basename(a.input)[:-3] if a.input else "pink-default"
        tag = f"{a.net}_{a.rule}_f{f:g}_{a.split}{res['n']}" + (f"_{inp}" if a.input else "")
        if not a.quick:
            np.save(os.path.join(OUTD, tag + ".npy"), k)
            np.save(os.path.join(OUTD, tag + "_agree.npy"), match)   # per-photo agree (critic, round 2)
        row = dict(net=a.net, rule=a.rule, f=f, split=a.split, input=inp, **res, params=params, shape=shp,
                   calc_seconds=round(t_calc, 1), total_seconds=round(time.time() - t0, 1),
                   photo_use=f"{a.split} photos, scoring only", when=datetime.datetime.now().isoformat(timespec="seconds"),
                   note=a.note, who=a.who, quick=a.quick)
        if a.quick:
            with open(os.path.join(HERE, "scratch.jsonl"), "a") as fh:
                fh.write(json.dumps(row) + "\n")
        else:
            add_row(row)
        print(json.dumps(row))
        del small
        if DEV == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
