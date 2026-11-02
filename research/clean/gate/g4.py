"""G4: variance collapse after LS refit, and REPAIR-style rescale. See CHILDREN-LOG G4."""
import json, os, sys
import numpy as np
import mlx.core as mx, mlx.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "coarse")); sys.path.insert(0, os.path.dirname(HERE))
from contract import mlp_io, orig_weights, cut_ls, SwiGLULayer, ROOT  # noqa: E402
from tree_lib import load_model, token_blocks  # noqa: E402
from r1 import block_ce  # noqa: E402


def silu(z): return z / (1 + np.exp(-z))


class Affine(nn.Module):
    def __init__(self, inner, a, b):
        super().__init__(); self.inner = inner; self.a = mx.array(a.astype(np.float32)); self.b = mx.array(b.astype(np.float32))
    def __call__(self, x): return self.inner(x) * self.a + self.b


def boot(d, n=2000):
    r = np.random.default_rng(0); m = np.array([d[r.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return [float(d.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]


model, tok = load_model(); cal = token_blocks(tok, "train", 64); dev = token_blocks(tok, "test", 40)
base = block_ce(model, dev); out = {}
for L in (12, 15, 26):
    X, _ = mlp_io(model, cal, L); G, U, D = orig_weights(model, L)
    Yo = (silu(X @ G.T) * (X @ U.T)) @ D.T
    G1n, U1n, _ = orig_weights(model, L + 1)
    for K in (48, 384):
        Gk, Uk, Dk, _ = cut_ls(model, L, K, X)
        Yc = (silu(X @ Gk.T) * (X @ Uk.T)) @ Dk.T
        so, sc = Yo.std(0), Yc.std(0) + 1e-9
        stats = {"std_ratio_med": float(np.median(sc / so)), "energy_ratio": float(np.sum(Yc ** 2) / np.sum(Yo ** 2))}
        layer = SwiGLULayer(Gk, Uk, Dk); orig = model.model.layers[L].mlp
        # downstream: gate pre-activation std at L+1 with the coarse layer in place
        X1o, _ = mlp_io(model, cal[:16], L + 1); model.model.layers[L].mlp = layer
        X1c, _ = mlp_io(model, cal[:16], L + 1)
        stats["next_gate_std_ratio_med"] = float(np.median((X1c @ G1n.T).std(0) / ((X1o @ G1n.T).std(0) + 1e-9)))
        ce = {"cut": block_ce(model, dev)}
        a_pc = so / sc; b_pc = Yo.mean(0) - Yc.mean(0) * a_pc
        model.model.layers[L].mlp = Affine(layer, a_pc, b_pc); ce["repair"] = block_ce(model, dev)
        g = np.sqrt(np.sum((Yo - Yo.mean(0)) ** 2) / np.sum((Yc - Yc.mean(0)) ** 2))
        model.model.layers[L].mlp = Affine(layer, np.full(len(so), g), np.zeros(len(so))); ce["global"] = block_ce(model, dev)
        model.model.layers[L].mlp = orig
        row = {**stats, "global_scale": float(g)}
        for k, v in ce.items():
            row[f"dce_{k}"] = float((v - base).mean())
        for k in ("repair", "global"):
            row[f"{k}_minus_cut"] = boot(ce[k] - ce["cut"]); row[f"{k}_recovered"] = float(1 - row[f"dce_{k}"] / row["dce_cut"])
        out[f"L{L}_K{K}"] = row; print(L, K, row, flush=True)
        json.dump(out, open(os.path.join(ROOT, "results", "clean-g4.json"), "w"), indent=1)
