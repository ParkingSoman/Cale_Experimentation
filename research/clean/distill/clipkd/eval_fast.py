"""Fast score of any image tower against the CLIP-KD teacher (2026-11-25). Goal: seconds, not minutes.

Reuses what score_clip.py saved once (out/<model>_test.npz): the teacher's class logits per photo, the photo ids,
labels, and each model's text classifier (1000 x 512, built from 80 prompts per class). So no text tower runs here.
A calculated network lives in the teacher's 512 output space, so it is scored with the TEACHER's classifier and scale.
Photos: the first n photos of the saved test set (one per class first, so n = 1000 covers every class once).
Photo use: test photos, scoring only.

Library:  from eval_fast import score;  score(visual)  -> dict(kl, top1, agree, seconds)
CLI check (reproduces score_clip.py numbers for the saved students, each with its own classifier and scale):
  PYTHONPATH=distill/clipkd/pylib curves/.venv/bin/python -I distill/clipkd/eval_fast.py --check kd --n 1000
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_clip import CLIP_MEAN, CLIP_STD, DEV, MODELS, OUT, build, net  # noqa: E402

_PH = None
_X = {}


def held_ids():
    """Held split as in neuron/coarse/r18.py: fit positions 8 and 9 of each class (2,000 photos).
    Stored class by class (like the test files), so teacher_ref puts one photo per class first."""
    ph = net.Photos("read")
    ids = []
    for c in range(1000):
        w = np.where(ph.lab == c)[0][0::2]
        ids += [w[8], w[9]]
    ids = np.array(ids)
    return ids, ph.lab[ids]


def make_held(models=("teacher", "kd")):
    """Once: teacher (and KD) image towers on the held photos, scored with the TEACHER's saved classifier and scale.
    Saves out/<model>_held.npz with the same fields as the test files. No text tower runs."""
    T = np.load(os.path.join(OUT, "teacher_test.npz"))
    cls, scale = torch.from_numpy(T["text_classifier"]).float(), float(T["logit_scale"])
    ids, lab = held_ids()
    for name in models:
        arch, fname = MODELS[name]
        m = build(arch, fname)
        X = photos(ids)
        emb = []
        with torch.no_grad():
            for i in range(0, len(ids), 200):
                emb.append(m.visual(X[i:i + 200].to(DEV).float()).float().cpu())
        emb = torch.cat(emb)
        lg = scale * F.normalize(emb, dim=-1) @ cls
        np.savez(os.path.join(OUT, f"{name}_held.npz"), emb=emb.numpy(), logits=lg.numpy(), labels=lab, photo_ids=ids,
                 logit_scale=scale, text_classifier=cls.numpy())
        print(name, "held top-1", float((lg.argmax(1).numpy() == lab).mean()))
        del m


def teacher_ref(split="test"):
    """split = "test" (scoring only) or "held" (the only split any choice may be tuned on)."""
    d = np.load(os.path.join(OUT, f"teacher_{split}.npz"))
    pid, lab, per_class_order = d["photo_ids"], d["labels"], None
    # photo order: one per class first (index 0, 5, 10, ... if per_class = 5), then the rest
    k = len(pid) // 1000
    order = np.concatenate([np.arange(j, len(pid), k) for j in range(k)]) if k > 1 else np.arange(len(pid))
    return dict(ids=pid[order], labels=lab[order], logits=torch.from_numpy(d["logits"][order]).float(),
                cls=torch.from_numpy(d["text_classifier"]).float(), scale=float(d["logit_scale"]), order=order)


def photos(ids):
    """Normalized photo tensor on the device. Cached in memory after the first call."""
    global _PH
    key = (int(ids[0]), len(ids))
    if key not in _X:
        if _PH is None:
            _PH = net.Photos("read")
        x = torch.from_numpy(_PH.raw(np.sort(ids)).copy()).permute(0, 3, 1, 2).float() / 255.0
        inv = np.argsort(np.argsort(ids))  # raw() reads in sorted order; put back in our order
        _X[key] = ((x[inv] - CLIP_MEAN) / CLIP_STD).half()
    return _X[key]


@torch.no_grad()
def score(visual, n=1000, cls=None, scale=None, bs=200, ref=None):
    t0 = time.time()
    ref = ref or teacher_ref()
    ids = ref["ids"][:n]
    X = photos(ids)
    cls = (ref["cls"] if cls is None else cls).to(DEV)
    scale = ref["scale"] if scale is None else scale
    lg = []
    for i in range(0, n, bs):
        e = visual(X[i:i + bs].to(DEV).float()).float()
        lg.append((scale * F.normalize(e, dim=-1) @ cls).cpu())
    lg = torch.cat(lg)
    pt = F.log_softmax(ref["logits"][:n], 1)
    ps = F.log_softmax(lg, 1)
    return dict(kl=float((pt.exp() * (pt - ps)).sum(1).mean()), top1=float((lg.argmax(1).numpy() == ref["labels"][:n]).mean()),
                agree=float((lg.argmax(1) == ref["logits"][:n].argmax(1)).float().mean()), n=n, seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default="kd", choices=list(MODELS))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--make_held", action="store_true")
    a = ap.parse_args()
    if a.make_held:
        make_held()
        return
    ref = teacher_ref()
    arch, fname = MODELS[a.check]
    m = build(arch, fname)
    d = np.load(os.path.join(OUT, f"{a.check}_test.npz"))
    t = time.time()
    photos(ref["ids"][:a.n])
    print(f"photos loaded {time.time() - t:.1f}s (once per process)")
    for rep in range(2):
        print(a.check, score(m.visual, n=a.n, cls=torch.from_numpy(d["text_classifier"]).float(), scale=float(d["logit_scale"]), ref=ref))


if __name__ == "__main__":
    main()
