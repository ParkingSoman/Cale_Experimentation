"""G6: route capture (circuit racing as an axiom). See CHILDREN-LOG G6.

  python3 research/clean/gate/g6.py --layers 12,15,26 --Ks 48,384
"""
import argparse, json, os, sys, time
import numpy as np
import mlx.core as mx, mlx.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.dirname(HERE))
from contract import mlp_io, orig_weights, ROOT  # noqa: E402
from tree_lib import load_model, token_blocks  # noqa: E402
from r1 import block_ce  # noqa: E402

FORBID = {6, 9, 21, 24}; BACK = 6


def silu(z): return z / (1.0 + np.exp(-z))


def boot(d, n=2000):
    r = np.random.default_rng(0); m = np.array([d[r.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return [float(d.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def ridge_ls(A, Y, lam=1e-4):
    M = A.T @ A; M += np.eye(M.shape[0]) * lam * np.trace(M) / M.shape[0]
    return np.linalg.solve(M, A.T @ Y)


class Stash(nn.Module):
    """Transparent wrapper on an upstream MLP that stores the hidden activations of chosen neurons."""
    def __init__(self, inner, G, U, box, key):
        super().__init__(); self.inner = inner; self.Gs = mx.array(G.astype(np.float32)); self.Us = mx.array(U.astype(np.float32))
        self.box, self.key = box, key
    def __call__(self, x):
        h = nn.silu(x @ self.Gs.T) * (x @ self.Us.T)
        if getattr(self, "mask0", False) and h.ndim == 3:     # no route write at block position 0 (attention sink)
            h = h * (mx.arange(h.shape[1]) > 0)[None, :, None]
        self.box[self.key] = h
        return self.inner(x)


class SkipLayer(nn.Module):
    def __init__(self, G, U, Dk, C_by_layer, box):
        super().__init__(); f = lambda a: mx.array(np.asarray(a, dtype=np.float32))
        self.G, self.U, self.D = f(G), f(U), f(Dk); self.C = {l: f(C) for l, C in C_by_layer.items()}; self.box = box
    def __call__(self, x):
        y = (nn.silu(x @ self.G.T) * (x @ self.U.T)) @ self.D.T
        for l, C in self.C.items():
            y = y + self.box[l] @ C.T
        return y


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="12,15,26"); ap.add_argument("--Ks", default="48,384")
    a = ap.parse_args(); layers = [int(x) for x in a.layers.split(",")]; assert not FORBID & set(layers)
    model, tok = load_model(); cal = token_blocks(tok, "train", 64); dev = token_blocks(tok, "test", 40)
    base = block_ce(model, dev)
    path = os.path.join(ROOT, "results", "clean-g6.json"); out = json.load(open(path)) if os.path.exists(path) else {}
    for L in layers:
        t0 = time.time()
        X, _ = mlp_io(model, cal, L); G, U, D = orig_weights(model, L)
        H = silu(X @ G.T) * (X @ U.T); Y = H @ D.T
        en = np.linalg.norm(H, axis=0) * np.linalg.norm(D, axis=0); order = np.argsort(-en)
        ups = {}
        for l in range(max(0, L - BACK), L):
            Xl, _ = mlp_io(model, cal, l); Gl, Ul, _ = orig_weights(model, l)
            ups[l] = (silu(Xl @ Gl.T) * (Xl @ Ul.T)).astype(np.float32)
        Aall = np.concatenate([ups[l] for l in sorted(ups)], 1); lab = [(l, j) for l in sorted(ups) for j in range(ups[l].shape[1])]
        print(f"L{L}: data ready [{time.time()-t0:.0f}s]", flush=True)
        for K in map(int, a.Ks.split(",")):
            res = {}
            def pick(keep, R):
                Hk = H[:, keep]; Dk = ridge_ls(Hk, Y); Yres = Y - Hk @ Dk
                sc = np.sum((Aall.T @ Yres) ** 2, 1) / (np.sum(Aall ** 2, 0) + 1e-9)
                return np.argsort(-sc)[:R]
            def fit(keep, routes):
                A = np.concatenate([H[:, keep], Aall[:, routes]], 1); W = ridge_ls(A, Y)
                return W[:len(keep)].T, W[len(keep):]          # Dk (d x k), C rows (R x d)
            def run_layer(layer):
                orig = model.model.layers[L].mlp; model.model.layers[L].mlp = layer
                ce = block_ce(model, dev); model.model.layers[L].mlp = orig; return ce
            # CUT
            keep = order[:K]; Dk = ridge_ls(H[:, keep], Y).T
            from contract import SwiGLULayer
            res["cut"] = run_layer(SwiGLULayer(G[keep], U[keep], Dk))
            # SKIP: K/2 kept + 3K/2 routes
            keep2 = order[:K // 2]; routes = pick(keep2, 3 * K // 2); Dk2, C = fit(keep2, routes)
            box = {}; by_l = {}
            for ri, r in enumerate(routes):
                by_l.setdefault(lab[r][0], []).append((lab[r][1], C[ri]))
            origs = {}
            for l, lst in by_l.items():
                js = [j for j, _ in lst]; Gl, Ul, _ = orig_weights(model, l)
                origs[l] = model.model.layers[l].mlp
                model.model.layers[l].mlp = Stash(origs[l], Gl[js], Ul[js], box, l)
            Cl = {l: np.stack([c for _, c in lst], 1) for l, lst in by_l.items()}
            res["skip"] = run_layer(SkipLayer(G[keep2], U[keep2], Dk2, Cl, box))
            for l, m in origs.items():
                model.model.layers[l].mlp = m
            # FOLD: K kept + R routes folded into upstream write columns
            for name, R in (("fold", K), ("fold3", 3 * K)):
                routes = pick(keep, R); DkF, CF = fit(keep, routes)
                saved = {}
                for ri, r in enumerate(routes):
                    l, j = lab[r]
                    if l not in saved:
                        saved[l] = model.model.layers[l].mlp.down_proj.weight
                    W = np.array(model.model.layers[l].mlp.down_proj.weight)
                    W[:, j] += CF[ri].astype(np.float32); model.model.layers[l].mlp.down_proj.weight = mx.array(W)
                res[name] = run_layer(SwiGLULayer(G[keep], U[keep], DkF))
                for l, Wt in saved.items():
                    model.model.layers[l].mlp.down_proj.weight = Wt
            row = {k: float((v - base).mean()) for k, v in res.items()}
            for k in ("skip", "fold", "fold3"):
                row[f"{k}_minus_cut"] = boot(res[k] - res["cut"])
            out[f"L{L}_K{K}"] = row; print(L, K, row, f"[{time.time()-t0:.0f}s]", flush=True)
            json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
