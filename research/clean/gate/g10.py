"""G10 (exploratory): attention fold. Routes = layer-L attention head outputs (input of o_proj, 576 numbers per
position). The route part is added to o_proj (0 extra stored numbers; attention is never coarsened, so this works
when every MLP is coarsened). Arms: cut, PLF (L-1 MLP routes), AF (L attention routes), AF+PLF. Sink guard everywhere.

  python3 research/clean/gate/g10.py --L 15 --K 48 [--code]
"""
import argparse, json, os, sys, time
import numpy as np
import mlx.core as mx, mlx.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from contract import mlp_io, orig_weights, ROOT, SwiGLULayer, SEQ  # noqa: E402
from tree_lib import load_model, token_blocks  # noqa: E402
from r1 import block_ce  # noqa: E402
from g6 import silu, boot, FORBID  # noqa: E402
from g7 import ridge, cv_ridge  # noqa: E402
from sink import SinkGuard  # noqa: E402


class Grab(nn.Module):
    def __init__(self, inner):
        super().__init__(); self.inner, self.x = inner, []
    def __call__(self, x):
        self.x.append(np.array(x).reshape(-1, x.shape[-1])); return self.inner(x)


class Lin(nn.Module):
    def __init__(self, W):
        super().__init__(); self.W = mx.array(np.asarray(W, dtype=np.float32))
    def __call__(self, x):
        return x @ self.W.T


def attn_in(model, blocks, L, batch=4):
    at = model.model.layers[L].self_attn; g = Grab(at.o_proj); at.o_proj = g
    for i in range(0, blocks.shape[0], batch):
        mx.eval(model(blocks[i:i + batch, :-1]))
    at.o_proj = g.inner; Z = np.concatenate(g.x, 0); keep = np.arange(Z.shape[0]) % SEQ != 0
    return Z[keep].astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--L", type=int, default=15); ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--code", action="store_true"); ap.add_argument("--mix", action="store_true"); ap.add_argument("--ranks", default=""); ap.add_argument("--tag", default="")
    a = ap.parse_args(); L, K = a.L, a.K; assert L not in FORBID and L >= 1
    model, tok = load_model(); cal = token_blocks(tok, "train", 64); dev = token_blocks(tok, "test", 40)
    if a.code:
        txt = open(os.path.join(ROOT, "research", "clean", "data", "code_train.txt")).read()
        ids = np.array(tok.encode(txt[:400000]), dtype=np.int32); nb = min(40, len(ids) // 513)
        dev = mx.array(ids[:nb * 513].reshape(nb, 513))
    if a.mix:   # calibration = 64 prose blocks + 64 code blocks from code_train.txt AFTER the dev part (chars 400k+)
        txt = open(os.path.join(ROOT, "research", "clean", "data", "code_train.txt")).read()[400000:]
        ids = np.array(tok.encode(txt[:1200000]), dtype=np.int32)[: 64 * 513]
        cal = mx.concatenate([cal, mx.array(ids.reshape(64, 513))], 0)
    base = block_ce(model, dev); t0 = time.time()
    X, _ = mlp_io(model, cal, L); G, U, D = orig_weights(model, L)
    H = (silu(X @ G.T) * (X @ U.T)).astype(np.float32); Y = (H @ D.T).astype(np.float32)
    en = np.linalg.norm(H, axis=0) * np.linalg.norm(D, axis=0); keep = np.argsort(-en)[:K]; Hk = H[:, keep]
    Xp, _ = mlp_io(model, cal, L - 1); Gp, Up, Dp = orig_weights(model, L - 1)
    Ap = (silu(Xp @ Gp.T) * (Xp @ Up.T)).astype(np.float32)
    Z = attn_in(model, cal, L); Wo = np.array(model.model.layers[L].self_attn.o_proj.weight, dtype=np.float64)
    orig_L = SwiGLULayer(G, U, D); orig_P = SwiGLULayer(Gp, Up, Dp)
    lay, layp, at = model.model.layers[L], model.model.layers[L - 1], model.model.layers[L].self_attn
    o_mlp, o_mlpp, o_op = lay.mlp, layp.mlp, at.o_proj

    def run(Dk, Cp=None, Ca=None):
        lay.mlp = SinkGuard(SwiGLULayer(G[keep], U[keep], Dk), orig_L)
        if Cp is not None:
            layp.mlp = SinkGuard(SwiGLULayer(Gp, Up, Dp + Cp.T), orig_P)
        if Ca is not None:
            at.o_proj = SinkGuard(Lin(Wo + Ca.T), Lin(Wo))
        ce = block_ce(model, dev); lay.mlp, layp.mlp, at.o_proj = o_mlp, o_mlpp, o_op; return ce
    res, lams = {}, {}
    W, lams["cut"] = cv_ridge(Hk, Y, (1e-4,)); res["cut"] = run(W.T)
    W, lams["plf"] = cv_ridge(np.concatenate([Hk, Ap], 1), Y); res["plf"] = run(W[:K].T, Cp=W[K:])
    W, lams["af"] = cv_ridge(np.concatenate([Hk, Z], 1), Y); res["af"] = run(W[:K].T, Ca=W[K:])
    W, lams["af_plf"] = cv_ridge(np.concatenate([Hk, Ap, Z], 1), Y)
    res["af_plf"] = run(W[:K].T, Cp=W[K:K + Ap.shape[1]], Ca=W[K + Ap.shape[1]:])
    def trunc(C, r):
        Uu, sv, Vt = np.linalg.svd(C, full_matrices=False); return (Uu[:, :r] * sv[:r]) @ Vt[:r]
    for r in [int(x) for x in a.ranks.split(",") if x]:
        Cp_r = trunc(W[K:K + Ap.shape[1]], r); Ca_r = trunc(W[K + Ap.shape[1]:], r)
        res[f"af_plf_r{r}"] = run(W[:K].T, Cp=Cp_r, Ca=Ca_r)
    if a.ranks:   # reduced-rank regression: project route writes onto the top-r directions of their fitted output
        Afr = np.concatenate([Ap, Z], 1); Cfull = W[K:]
        Yh = Afr[::4] @ Cfull; _, _, Vt = np.linalg.svd(Yh - Yh.mean(0), full_matrices=False)
        for r in [int(x) for x in a.ranks.split(",") if x]:
            P = Vt[:r].T @ Vt[:r]; Cr = Cfull @ P
            res[f"af_plf_rrr{r}"] = run(W[:K].T, Cp=Cr[:Ap.shape[1]], Ca=Cr[Ap.shape[1]:])
    row = {k: float((v - base).mean()) for k, v in res.items()}; row["lam"] = lams
    for k in res:
        if k != "cut":
            row[k + "_minus_cut"] = boot(res[k] - res["cut"]); row[k + "_recovered"] = float(1 - row[k] / row["cut"])
    path = os.path.join(ROOT, "results", "clean-g10.json"); out = json.load(open(path)) if os.path.exists(path) else {}
    out[f"L{L}_K{K}{'_code' if a.code else ''}{'_mix' if a.mix else ''}{a.tag}"] = row; json.dump(out, open(path, "w"), indent=1)
    print(L, K, "code" if a.code else "prose", "mixcal" if a.mix else "", {k: round(row[k], 4) for k in res}, {k: round(row[k + "_recovered"], 3) for k in res if k != "cut"},
          f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
