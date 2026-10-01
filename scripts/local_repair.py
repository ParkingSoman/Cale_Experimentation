"""Layer-local test of calibration-only repairs for MLP width cuts (bf16, layer by layer).

For selected layers and widths, fit on CALIBRATION samples, evaluate on HELD-OUT samples:
  none     keep the top-n neurons only
  lin-r64  add a rank-64 linear correction predicted from the MLP input x
           (reduced-rank ridge regression of the cut error on x; 2 x 3840 x 64 numbers per layer)
  lin-full same, full rank (3840 x 3840), to show the ceiling of a linear-in-x correction
  refit    re-solve the kept down_proj columns by least squares (upper bound; breaks strict nesting)
"""
import sys

import mlx.core as mx
import numpy as np
from mlx_vlm.models.gemma4 import language as g4

sys.path.insert(0, ".")
from nest.io import read_jsonl
from nest.levels import load_stats, scores
from nest.stream import LayerwiseRunner

LAYERS = [2, 11, 24, 40]
WIDTHS = [13824, 11520]
RANK = 64
REFIT_WIDTH = 11520
LAMBDA = 1e-3
order = np.argsort(-scores(load_stats(), "wanda"), axis=1)
CUR = {"i": None, "phase": "fit"}
acc = {}       # (layer, width) -> dict of Gram matrices (float64 numpy)
fits = {}      # (layer, width) -> fitted correction matrices
errs = {}      # (layer, width, method) -> list of relative errors


def mlp_call(self, x):
    a = g4.geglu(self.gate_proj(x), self.up_proj(x))
    i = CUR["i"]
    y = self.down_proj(a)
    if i not in LAYERS:
        return y
    X = x[0].astype(mx.float32)
    Y = y[0].astype(mx.float32)
    A = a[0].astype(mx.float32)
    W = self.down_proj.weight.astype(mx.float32)  # (H, I)
    for n in WIDTHS:
        keep = mx.array(order[i][:n])
        Ak = A[:, keep]
        Wk = W[:, keep]
        base = Ak @ Wk.T
        E = Y - base
        key = (i, n)
        if CUR["phase"] == "fit":
            g = acc.setdefault(key, {})
            terms = [("XtX", X.T @ X), ("XtE", X.T @ E)]
            if n == REFIT_WIDTH:
                terms += [("AtA", Ak.T @ Ak), ("AtY", Ak.T @ Y)]
            for name, val in terms:
                mx.eval(val)
                g[name] = g.get(name, 0) + np.array(val, dtype=np.float32 if name.startswith("At") else np.float64)
        else:
            f = fits[key]
            outs = {"none": base,
                    "lin-r64": base + (X @ mx.array(f["U"])) @ mx.array(f["V"]).T,
                    "lin-full": base + X @ mx.array(f["B"])}
            if "D" in f:
                outs["refit"] = Ak @ mx.array(f["D"])
            yn = mx.linalg.norm(Y)
            for m, o in outs.items():
                e = (mx.linalg.norm(o - Y) / yn).item()
                errs.setdefault((i, n, m), []).append(e)
    return y


def solve():
    for key, g in acc.items():
        H = g["XtX"].shape[0]
        reg = LAMBDA * np.trace(g["XtX"]) / H
        B = np.linalg.solve(g["XtX"] + reg * np.eye(H), g["XtE"])          # (H, H): E ≈ X B
        # Reduced-rank: best rank-r approximation of the fitted map in the X-weighted norm.
        L = np.linalg.cholesky(g["XtX"] + reg * np.eye(H))
        Uw, S, Vt = np.linalg.svd(L.T @ B, full_matrices=False)
        U = np.linalg.solve(L.T, Uw[:, :RANK] * S[:RANK])                   # (H, r)
        V = Vt[:RANK].T                                                     # (H, r)
        fits[key] = {"B": B.astype(np.float32), "U": U.astype(np.float32), "V": V.astype(np.float32)}
        if "AtA" in g:
            n = g["AtA"].shape[0]
            reg2 = LAMBDA * np.trace(g["AtA"]) / n
            g["AtA"][np.diag_indices(n)] += reg2
            fits[key]["D"] = np.linalg.solve(g["AtA"], g["AtY"]).astype(np.float32)   # Y ≈ A_keep D
            del g["AtA"], g["AtY"]
        print("solved", key, flush=True)


g4.MLP.__call__ = mlp_call
calib = [r for i, r in enumerate(read_jsonl("data/calib.jsonl")) if i % 2 == 0]   # 160 samples
held = [r for r in read_jsonl("data/heldout.jsonl") if int(r["id"].split("-")[-1]) % 5 == 0]

for phase, recs in (("fit", calib), ("eval", held)):
    CUR["phase"] = phase
    runner = LayerwiseRunner("models/gemma-4-12B-it-bf16")
    runner.prepare(recs)
    Cls = runner.lm.layers[0].__class__
    if not hasattr(Cls, "_orig_call"):
        Cls._orig_call = Cls.__call__

        def layer_call(self, *a, **k):
            CUR["i"] = self.layer_idx
            return Cls._orig_call(self, *a, **k)
        Cls.__call__ = layer_call
    # Only layers up to the last tested one are needed.
    runner.lm.layers = runner.lm.layers[: max(LAYERS) + 1]
    runner.run()
    print(f"{phase} pass done on {len(recs)} samples", flush=True)
    if phase == "fit":
        solve()
        acc.clear()

print(f"\nHeld-out relative MLP output error ({len(held)} samples). Lower is better.")
print("layer  width |  none   lin-r64  lin-full  refit")
for i in LAYERS:
    for n in WIDTHS:
        r = {m: np.mean(errs[(i, n, m)]) if (i, n, m) in errs else float("nan") for m in ("none", "lin-r64", "lin-full", "refit")}
        print(f"{i:5d} {n:6d} | {r['none']:.3f}   {r['lin-r64']:.3f}    {r['lin-full']:.3f}    {r['refit']:.3f}")
