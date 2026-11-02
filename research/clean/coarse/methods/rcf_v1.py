"""Route-capture fold (RCF), family "route-capture" (CHILDREN-LOG G6-G11, ESSAY-IDEAS I13). Base model + public text.

Keep the cut's top-K neurons at layer L. The removed neurons' output is captured by the routes that already fire just
before them: layer L's attention head outputs (input of o_proj) and layer L-1's MLP neurons. One CV-ridge LS of
[kept L hiddens, L-1 hiddens, L attention head outputs] -> original layer-L output. The route parts are ADDED to
layer L's o_proj and layer L-1's down_proj (no new stored numbers for one coarse model; for a nested dial they are
deltas). Calibration: `cal` prose blocks + the same number of code blocks from code_train.txt chars 400k+.
Sink guard: every changed module returns its original output at block position 0.

CONTRACT EXTENSION: the returned layer has `.extra = [(owner_module, attr_name, new_module), ...]`. The evaluator must
set owner.attr = new_module before scoring and restore it after. PLF is skipped when L - 1 == 0.
"""
import os, sys
import numpy as np
import mlx.core as mx, mlx.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import orig_weights, SwiGLULayer, SEQ, ROOT  # noqa: E402

METHOD = {"name": "rcf_v1", "family": "route-capture", "minutes": 2.0}
N_CAL = 64


def silu(z):
    return z / (1.0 + np.exp(-z))


class Lin(nn.Module):
    def __init__(self, W):
        super().__init__(); self.W = mx.array(np.asarray(W, dtype=np.float32))
    def __call__(self, x):
        return x @ self.W.T


class SinkGuard(nn.Module):
    def __init__(self, new, orig):
        super().__init__(); self.new, self.orig = new, orig
    def __call__(self, x):
        y = self.new(x)
        if x.ndim == 3:
            y = mx.concatenate([self.orig(x[:, :1]), y[:, 1:]], axis=1)
        return y
    def units(self, x):
        return self.new.units(x)
    def n_params(self):
        return self.new.n_params()


class _Grab(nn.Module):
    def __init__(self, inner):
        super().__init__(); self.inner, self.x = inner, []
    def __call__(self, x):
        self.x.append(np.array(x).reshape(-1, x.shape[-1]).astype(np.float32)); return self.inner(x)


def _capture(model, blocks, mods, batch=4):
    gs = {}
    for k, (par, at) in mods.items():
        gs[k] = _Grab(getattr(par, at)); setattr(par, at, gs[k])
    for i in range(0, blocks.shape[0], batch):
        mx.eval(model(blocks[i:i + batch, :-1]))
    out = {}
    for k, (par, at) in mods.items():
        setattr(par, at, gs[k].inner); X = np.concatenate(gs[k].x, 0); out[k] = X[np.arange(len(X)) % SEQ != 0]
    return out


def _ridge(A, Y, lam):
    M = (A.T @ A).astype(np.float64); M += np.eye(M.shape[0]) * lam * np.trace(M) / M.shape[0]
    return np.linalg.solve(M, (A.T @ Y).astype(np.float64))


def _cv_ridge(A, Y, lams=(1e-3, 1e-2, 1e-1, 3e-1)):
    nb = A.shape[0] // 2; best = (np.inf, lams[0])
    for lam in lams:
        e = 0.0
        for tr, te in ((slice(0, nb), slice(nb, None)), (slice(nb, None), slice(0, nb))):
            W = _ridge(A[tr], Y[tr], lam); e += float(np.sum((Y[te] - A[te] @ W) ** 2))
        if e < best[0]:
            best = (e, lam)
    return _ridge(A, Y, best[1])


def build(model, tok, L, K, cal, log=print):
    txt = open(os.path.join(ROOT, "research", "clean", "data", "code_train.txt")).read()[400000:]
    ids = np.array(tok.encode(txt[:1200000]), dtype=np.int32)[: N_CAL * 513]
    blocks = mx.concatenate([cal[:N_CAL], mx.array(ids.reshape(N_CAL, 513))], 0)
    lay = model.model.layers[L]; plf = L - 1 >= 1
    mods = {"x": (lay, "mlp"), "z": (lay.self_attn, "o_proj")}
    if plf:
        mods["xp"] = (model.model.layers[L - 1], "mlp")
    cap = _capture(model, blocks, mods)
    G, U, D = orig_weights(model, L)
    X = cap["x"].astype(np.float64); H = silu(X @ G.T) * (X @ U.T); Y = (H @ D.T).astype(np.float32)
    en = np.linalg.norm(H, axis=0) * np.linalg.norm(D, axis=0); keep = np.sort(np.argsort(-en)[:K])
    feats = [H[:, keep].astype(np.float32)]
    if plf:
        Gp, Up, Dp = orig_weights(model, L - 1); Xp = cap["xp"].astype(np.float64)
        feats.append((silu(Xp @ Gp.T) * (Xp @ Up.T)).astype(np.float32))
    feats.append(cap["z"])
    W = _cv_ridge(np.concatenate(feats, 1), Y)
    Dk = W[:K].T; i = K; extra = []
    if plf:
        Cp = W[i:i + Gp.shape[0]]; i += Gp.shape[0]
        extra.append((model.model.layers[L - 1], "mlp", SinkGuard(SwiGLULayer(Gp, Up, Dp + Cp.T), SwiGLULayer(Gp, Up, Dp))))
    Wo = np.array(lay.self_attn.o_proj.weight, dtype=np.float64); Ca = W[i:]
    extra.append((lay.self_attn, "o_proj", SinkGuard(Lin(Wo + Ca.T), Lin(Wo))))
    layer = SinkGuard(SwiGLULayer(G[keep], U[keep], Dk), SwiGLULayer(G, U, D))
    layer.extra = extra
    log(f"  rcf_v1 L{L} K{K}: {len(extra)} extra modules")
    return layer
