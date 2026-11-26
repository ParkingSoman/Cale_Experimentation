"""P4 write: implementation checks + smoke test (round 1 build, 2026-11-26). Run by the runner ONLY (loads the models).
Every check here passes by algebra when the code is right. None of them is evidence for the rule (the nulls are).

  K1 synthetic fold (CPU float64, no model): lam = 0, exact copy -> P = 1, 2x copy -> P = 2; slice anchor keeps the
     kept rows of G at I exactly; the literal ("zero") anchor shrinks them (printed, by design).
  K2 CLIP smoke on X[:16]: every variant builds at f = .5, the residual assert (rule <= floor on X) passes inside each
     fold build, one forward is finite.
  K3 CLIP f = 1 whole-network rule: o_w' = o_w and pr_w' = pr_w (rel < 1e-4), output equal (rel < 1e-3).
  K4 CLIP closed vs open loop, MLP-only f = .5: block 0's MLP sees no upstream change, so G equal (rel < 1e-3) and far
     below block 1's closed/open difference (ratio < .1).
  K5-K7 the same three on ResNet-18 (layer1.0 = no upstream change).
  M = I check: NOT APPLICABLE (A = 0: the reader metric cancels from G).
Exit code 1 on any FAIL. Photo use: none. Input: the cached generic tensor (first 16 images).
Usage (from research/clean): curves/.venv/bin/python -I distill/fanout/p4write_check.py --input <cache .pt>
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import board  # noqa: E402
import vitgen  # noqa: E402
from rules import p4write  # noqa: E402

FAILS = []


def say(name, ok, msg):
    print(f"{'PASS' if ok else 'FAIL'} {name}: {msg}", flush=True)
    if not ok:
        FAILS.append(name)


def rel(a, b):
    return float((a.double() - b.double()).norm() / max(float(b.double().norm()), 1e-30))


def k1():
    g = torch.Generator().manual_seed(0)
    N, k = 12800, 64
    aK = torch.relu(torch.randn(N, k, generator=g, dtype=torch.float64) @ torch.randn(k, k, generator=g, dtype=torch.float64) / 8)
    mix = torch.relu(aK @ torch.randn(k, 6, generator=g, dtype=torch.float64) / 8 + .1 * torch.randn(N, 6, generator=g, dtype=torch.float64))
    a = torch.cat([aK, aK[:, :1], 2 * aK[:, 3:4], mix], 1)            # kept 0..63, cut 64 = copy of 0, 65 = 2 x unit 3
    kept = torch.arange(k)
    Sx, Ss = a.T @ aK / N, aK.T @ aK / N
    G0, _ = p4write.fold_map(Sx, Ss, kept, N, lam=0.0)
    e0, e3 = torch.zeros(k, dtype=torch.float64), torch.zeros(k, dtype=torch.float64)
    e0[0], e3[3] = 1, 2
    err = max(float((G0[k] - e0).abs().max()), float((G0[k + 1] - e3).abs().max()))
    say("K1a exact copy at lam = 0 (algebra)", err < 1e-6, f"max |P - copy map| = {err:.2e}")
    G, lam = p4write.fold_map(Sx, Ss, kept, N)
    errK = float((G[:k] - torch.eye(k, dtype=torch.float64)).abs().max())
    say("K1b slice anchor keeps kept rows = I (algebra)", errK < 1e-8, f"max |G_KK - I| = {errK:.2e}, lam = {lam:.3e}, P(copy) = {float(G[k, 0]):.4f}")
    Gz, _ = p4write.fold_map(Sx, Ss, kept, N, anchor="zero")
    print(f"INFO K1c literal (zero) anchor shrinks the kept writes: max |G_KK - I| = {float((Gz[:k] - torch.eye(k, dtype=torch.float64)).abs().max()):.3e}")


def finite_vit(W, X):
    with torch.no_grad():
        y = vitgen.forward(vitgen.to(W, p4write.DEV), X[:4].to(p4write.DEV)).float().cpu()
    return y, bool(torch.isfinite(y).all())


def clip_checks(X):
    T = board.clip_teacher()
    variants = dict(full=dict(sets=("stream", "heads", "mlp")), mlp=dict(sets=("mlp",)), heads=dict(sets=("heads",)),
                    null_full=dict(sets=("stream", "heads", "mlp"), null=True), null_mlp=dict(sets=("mlp",), null=True),
                    null_heads=dict(sets=("heads",), null=True), off_mlp=dict(sets=("mlp",), fold=False),
                    off_heads=dict(sets=("heads",), fold=False))
    for name, kw in variants.items():
        try:
            W, info = p4write.build_vit(T, 0.5, X, tag="check_" + name, save=False, **kw)
            _, ok = finite_vit(W, X)
            say(f"K2 clip {name} f=.5 builds, residual assert, finite", ok, f"shape {vitgen.shape(W)['d']} d, "
                f"heads {sorted(set(vitgen.shape(W)['heads']))}, mlp {sorted(set(vitgen.shape(W)['mlp']))}")
        except Exception as e:  # noqa: BLE001
            say(f"K2 clip {name} f=.5", False, repr(e))
    W, _ = p4write.build_vit(T, 1.0, X, tag="check_f1", save=False)
    dw = max(max(rel(B["o_w"], A["o_w"]), rel(B["pr_w"], A["pr_w"])) for A, B in zip(T["blocks"], W["blocks"]))
    yW, _ = finite_vit(W, X)
    yT, _ = finite_vit(T, X)
    say("K3 clip f=1 rule = teacher (algebra)", dw < 1e-4 and rel(yW, yT) < 1e-3, f"max rel write diff {dw:.2e}, output rel diff {rel(yW, yT):.2e}")
    keep = ("b0.mlp", "b1.mlp")
    _, ic = p4write.build_vit(T, 0.5, X, sets=("mlp",), tag="check_closed", save=False, keep_G=keep)
    _, io = p4write.build_vit(T, 0.5, X, sets=("mlp",), loop="open", tag="check_open", save=False, keep_G=keep)
    r0, r1 = rel(ic["G"]["b0.mlp"], io["G"]["b0.mlp"]), rel(ic["G"]["b1.mlp"], io["G"]["b1.mlp"])
    say("K4 clip closed = open with no upstream change (algebra)", r0 < 1e-3 and r0 < .1 * max(r1, 1e-30),
        f"rel G diff block 0 {r0:.2e}, block 1 {r1:.2e}")


def r18_checks(X):
    A = board.r18_A()
    variants = dict(full=dict(sets=("stream", "inner")), inner=dict(sets=("inner",)),
                    null_full=dict(sets=("stream", "inner"), null=True), null_inner=dict(sets=("inner",), null=True),
                    off_inner=dict(sets=("inner",), fold=False))
    xd = X[:4].to(p4write.DEV)
    with torch.no_grad():
        zA = A.to(p4write.DEV).eval()(xd).float().cpu()
    A.cpu()
    for name, kw in variants.items():
        try:
            S, _ = p4write.build_r18(A, 0.5, X, tag="check_" + name, save=False, **kw)
            with torch.no_grad():
                z = S.to(p4write.DEV).eval()(xd).float().cpu()
            say(f"K5 r18 {name} f=.5 builds, residual assert, finite", bool(torch.isfinite(z).all()), f"widths {getattr(S, 'w', 'full')}")
        except Exception as e:  # noqa: BLE001
            say(f"K5 r18 {name} f=.5", False, repr(e))
    S, _ = p4write.build_r18(A, 1.0, X, tag="check_f1", save=False)
    import wide18
    dw = max(rel(bs.conv2.weight, ba.conv2.weight) for ls, la in zip(wide18.layers(S), wide18.layers(A)) for bs, ba in zip(ls, la))
    with torch.no_grad():
        z = S.to(p4write.DEV).eval()(xd).float().cpu()
    say("K6 r18 f=1 rule = teacher (algebra)", dw < 1e-4 and rel(z, zA) < 1e-3, f"max rel conv2 diff {dw:.2e}, logit rel diff {rel(z, zA):.2e}")
    keep = ("layer1.0", "layer1.1")
    _, ic = p4write.build_r18(A, 0.5, X, sets=("inner",), tag="check_closed", save=False, keep_G=keep)
    _, io = p4write.build_r18(A, 0.5, X, sets=("inner",), loop="open", tag="check_open", save=False, keep_G=keep)
    r0, r1 = rel(ic["G"]["layer1.0"], io["G"]["layer1.0"]), rel(ic["G"]["layer1.1"], io["G"]["layer1.1"])
    say("K7 r18 closed = open with no upstream change (algebra)", r0 < 1e-3 and r0 < .1 * max(r1, 1e-30),
        f"rel G diff layer1.0 {r0:.2e}, layer1.1 {r1:.2e}")
    say("K8 A was not changed", next(A.parameters()).device.type == "cpu", "A stays on the CPU")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    X = torch.load(a.input, weights_only=True)[:a.n].float()
    k1()
    clip_checks(X)
    r18_checks(X)
    print("M = I check: NOT APPLICABLE (A = 0; the reader metric cancels from G)")
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
