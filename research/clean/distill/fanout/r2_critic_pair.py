"""CRITIC round 2: paired comparison of two rules on the SAME photos (a reading check, never on the board).

Object (picked by): two whole small networks built by rules A and B at one size f (picked by the claim under test).
Photo use: test photos (quick set, every 5th of the 1,000) for scoring only. The temperature column fits ONE scalar
on these test photos: a labelled calibration reading, never a rule input.

Prints per rule: KL, agree, KL after the best single temperature (KL_T), T. Then the paired lines:
  discordant agree counts (A right & B wrong, B right & A wrong), McNemar z,
  paired KL difference B - A with its SE.
Usage (from research/clean, through torchq):
  .../python -I distill/fanout/r2_critic_pair.py --net clip --a noisecut --b p4write_mlp --f 0.85 [--input ...] [--stride 5]
"""
import argparse
import importlib
import inspect
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import board  # noqa: E402  (has a __main__ guard)


@torch.no_grad()
def logits(net, small, stride, ref_cache={}):
    if net == "clip":
        sys.path.insert(0, os.path.join(board.DIST, "clipkd"))
        import eval_fast
        import vitgen
        if "clip" not in ref_cache:
            ref = eval_fast.teacher_ref("test")
            sel = np.arange(min(1000, len(ref["ids"])))[::stride]
            X = eval_fast.photos(ref["ids"][sel])
            ref_cache["clip"] = (ref, sel, X)
        ref, sel, X = ref_cache["clip"]
        Wd = vitgen.to(small, board.DEV)
        cls = ref["cls"].to(board.DEV)
        out = []
        for i in range(0, len(X), 100):
            e = vitgen.forward(Wd, X[i:i + 100].to(board.DEV).float()).float()
            out.append((ref["scale"] * F.normalize(e, dim=-1) @ cls).cpu())
        return torch.cat(out), ref["logits"][sel].float()
    x, _ = board.r18_x("test", 1000, stride)
    zA = torch.load(os.path.join(board.CACHE, f"r18_A_test1000s{stride}.pt" if stride > 1 else "r18_A_test1000.pt"),
                    weights_only=True) if os.path.exists(os.path.join(board.CACHE, f"r18_A_test1000s{stride}.pt" if stride > 1 else "r18_A_test1000.pt")) else board.r18_logits(board.r18_A(), x)
    return board.r18_logits(small, x), zA.float()


def best_T(zt, z):
    best = (1e9, 1.0)
    for T in np.exp(np.linspace(np.log(.1), np.log(10), 61)):
        k = float(board.kl_rows(zt, z / T).mean())
        best = min(best, (k, float(T)))
    return best


def build(net, rule, f, X):
    m = importlib.import_module(f"rules.{rule}")
    fn = getattr(m, "vit" if net == "clip" else "r18")
    big = board.clip_teacher() if net == "clip" else board.r18_A()
    return fn(big, f, X=X) if "X" in inspect.signature(fn).parameters and X is not None else fn(big, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True, help="comma list allowed: each compared with A")
    ap.add_argument("--f", default="0.95")
    ap.add_argument("--input", default="")
    ap.add_argument("--stride", type=int, default=5)
    a = ap.parse_args()
    X = torch.load(a.input, weights_only=True).float() if a.input else None
    for f in [float(v) for v in a.f.split(",")]:
        res = {}
        for r in [a.a] + a.b.split(","):
            z, zt = logits(a.net, build(a.net, r, f, X), a.stride)
            k = board.kl_rows(zt, z)
            ag = (z.argmax(1) == zt.argmax(1)).numpy()
            kT, T = best_T(zt, z)
            res[r] = (k.numpy(), ag)
            print(json.dumps(dict(net=a.net, f=f, rule=r, n=len(ag), kl=round(float(k.mean()), 4), agree=float(ag.mean()),
                                  kl_T=round(kT, 4), T=round(T, 3))), flush=True)
            if board.DEV == "mps":
                torch.mps.empty_cache()
        kA, gA = res[a.a]
        for r in a.b.split(","):
            kB, gB = res[r]
            n10, n01 = int((gA & ~gB).sum()), int((~gA & gB).sum())
            d = kB - kA
            z = (n01 - n10) / max(1, np.sqrt(n01 + n10))
            print(f"PAIR f={f} A={a.a} B={r}: A-only right {n10}, B-only right {n01}, McNemar z {z:+.2f}; "
                  f"KL B-A {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}", flush=True)


if __name__ == "__main__":
    main()
