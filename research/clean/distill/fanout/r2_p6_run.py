"""P6 round 2 batch runner (one process, teacher loaded once). Builds stream-only students with rules/r2_p6_core.py,
scores them on the quick photo set (200 test photos, every 5th class), appends rows to fanout/scratch.jsonl.
Also: open-loop reader error per basis (diag) and the planted T+ test (unread high-variance direction).
Object (picked by): the stream basis Q (P6 round-1 revised proposal). Input: pink X (cache/input_pink_n256_s1.pt).
--photo: LABELLED READING CHECK, the basis moments come from 256 held photos; such rows carry input 'held256-CHECK'
and never go to board.json.
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
from rules import r2_p6_core as core  # noqa: E402

dt = torch.float64


@torch.no_grad()
def score_fwd(W, fwd, n=1000, stride=5, bs=100):
    """board.clip_score with a custom forward (same photos, same reference, same formulas)."""
    import numpy as np
    import torch.nn.functional as F
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "clipkd"))
    import eval_fast
    import vitgen
    ref = eval_fast.teacher_ref("test")
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
    lab = ref["labels"][:n]
    return dict(kl=float(k.mean()), top1=float((lg.argmax(1).numpy() == lab).mean()),
                agree=float((lg.argmax(1) == zt.argmax(1)).float().mean()), n=n), k.numpy()


def read_err(T, X, Q):
    m = core.site_moments(T, X)
    Gs = core.readers(T)
    D = Q.shape[0]
    Pp = torch.eye(D, dtype=dt) - Q @ Q.T
    out = []
    for M, G in zip(m["M"], Gs):
        out.append(float(torch.trace(G @ Pp @ M @ Pp) / torch.trace(G @ M)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="kind:dmode:f1/f2,...")
    ap.add_argument("--photo", action="store_true")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--plant", action="store_true")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--stride", type=int, default=5)
    a = ap.parse_args()
    T = board.clip_teacher()
    if a.photo:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "clipkd"))
        import eval_fast
        ids = eval_fast.teacher_ref("held")["ids"][1000:2000][::4][:256]
        X = eval_fast.photos(ids).float()
        inp = "held256-CHECK"
    else:
        X = torch.load(os.path.join(HERE, "cache", "input_pink_n256_s1.pt"), weights_only=True).float()
        inp = "input_pink_n256_s1"
    D = T["cls"].numel()
    if a.plant:
        g = torch.Generator().manual_seed(7)
        u = core._orth_centred(torch.randn(D, 1, generator=g, dtype=dt))[:, 0]
        C = core.site_moments(T, X)["M"].sum(0)
        lam = 10 * float(torch.linalg.eigvalsh(C)[-1]) / C.shape[0] * 1.0
        lam = 10 * float(torch.linalg.eigvalsh(C)[-1]) / 25
        for kind in ("pca", "pcan", "read", "readG", "rand"):
            for f in (0.85, 0.5):
                k = round(f * D)
                Q, _ = core.basis(T, X, k, kind, plant=(u, lam))
                print(f"[T+] Object (picked by): planted unread direction u (random centred, seed 7; lam = 10x top site eig) "
                      f"basis {kind} f={f}: kept share of u |Q^T u|^2 = {float((Q.T @ u).pow(2).sum()):.3f} "
                      f"(pass = drop: < .05; chance = {k / D:.2f})", flush=True)
    for c in a.cfg.split(","):
        kind, dmode, fs = c.split(":")
        for f in [float(v) for v in fs.split("/")]:
            t0 = time.time()
            if kind.startswith("ps-"):
                from rules import r2_p6_persite as ps
                W = ps.build(T, f, X, kind[3:])
                res, _ = score_fwd(W, ps.forward, a.n, a.stride)
                res["params"] = ps.n_params(W)
            else:
                W = core.build(T, f, X, kind, dmode)
            if a.diag and not kind.startswith("ps-"):
                Q, _ = core.basis(T, X, round(f * D), kind)
                e = read_err(T, X, Q)
                print(f"[diag] {kind} f={f}: open-loop reader error per site mean {sum(e)/len(e):.3f} "
                      f"first4 {' '.join(f'{v:.2f}' for v in e[:4])} last4 {' '.join(f'{v:.2f}' for v in e[-4:])}", flush=True)
            if not kind.startswith("ps-"):
                res, _ = board.clip_score(W, a.n, "test", stride=a.stride)
            row = dict(net="clip", rule=f"r2_p6_{kind}_{dmode}", f=f, split="test", input=inp, **res,
                       calc_seconds=round(time.time() - t0, 1), photo_use="test photos scoring only" + (
                           "; HELD PHOTOS IN THE BASIS (labelled reading check)" if a.photo else ""),
                       when=datetime.datetime.now().isoformat(timespec="seconds"), who="P6", quick=True, note="r2 P6 runner")
            print(f"ROW {kind}/{dmode} f={f} [{inp}] KL {res['kl']:.3f} agree {100 * res['agree']:.1f}% top1 {100 * res['top1']:.1f}%", flush=True)
            with open(os.path.join(HERE, "scratch.jsonl"), "a") as fh:
                fh.write(json.dumps(row) + "\n")
            del W
            if board.DEV == "mps":
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()
