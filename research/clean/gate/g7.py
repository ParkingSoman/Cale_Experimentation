"""G7: route capture with the sink guard (see CHILDREN-LOG G7).

  python3 research/clean/gate/g7.py --layers 3,12,15,18,26 --K 48
"""
import argparse, json, os, sys, time
import numpy as np
import mlx.core as mx, mlx.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from contract import mlp_io, orig_weights, ROOT, SwiGLULayer  # noqa: E402
from tree_lib import load_model, token_blocks  # noqa: E402
from r1 import block_ce  # noqa: E402
from g6 import silu, boot, FORBID, Stash, SkipLayer  # noqa: E402
from sink import SinkGuard  # noqa: E402

BACK = 6


def ridge(A, Y, lam):
    M = (A.T @ A).astype(np.float64); M += np.eye(M.shape[0]) * lam * np.trace(M) / M.shape[0]
    return np.linalg.solve(M, (A.T @ Y).astype(np.float64))


def cv_ridge(A, Y, lams=(1e-3, 1e-2, 1e-1, 3e-1)):
    nb = A.shape[0] // 2; best = (np.inf, None)
    for lam in lams:
        e = 0.0
        for tr, te in ((slice(0, nb), slice(nb, None)), (slice(nb, None), slice(0, nb))):
            W = ridge(A[tr], Y[tr], lam); e += float(np.sum((Y[te] - A[te] @ W) ** 2))
        if e < best[0]:
            best = (e, lam)
    return ridge(A, Y, best[1]), best[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="3,12,15,18,26"); ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--mults", default="10,30"); ap.add_argument("--tag", default="")
    a = ap.parse_args(); layers = [int(x) for x in a.layers.split(",")]; assert not FORBID & set(layers); K = a.K
    model, tok = load_model(); cal = token_blocks(tok, "train", 64); dev = token_blocks(tok, "test", 40)
    base = block_ce(model, dev)
    path = os.path.join(ROOT, "results", "clean-g7.json"); out = json.load(open(path)) if os.path.exists(path) else {}
    mults = [int(x) for x in a.mults.split(",")]
    for L in layers:
        t0 = time.time()
        X, _ = mlp_io(model, cal, L); G, U, D = orig_weights(model, L)
        H = (silu(X @ G.T) * (X @ U.T)).astype(np.float32); Y = (H @ D.T).astype(np.float32)
        en = np.linalg.norm(H, axis=0) * np.linalg.norm(D, axis=0); order = np.argsort(-en)
        lays = list(range(max(0, L - BACK), L)); acts, lab = [], []
        for l in lays:
            Xl, _ = mlp_io(model, cal, l); Gl, Ul, _ = orig_weights(model, l)
            acts.append((silu(Xl @ Gl.T) * (Xl @ Ul.T)).astype(np.float32)); lab += [(l, j) for j in range(Gl.shape[0])]
        Aall = np.concatenate(acts, 1); del acts
        orig_L = SwiGLULayer(G, U, D)

        def run(layer):
            o = model.model.layers[L].mlp; model.model.layers[L].mlp = SinkGuard(layer, orig_L)
            ce = block_ce(model, dev); model.model.layers[L].mlp = o; return ce

        def rank_routes(keep):
            Hk = H[:, keep]; W = ridge(Hk, Y, 1e-4); Yres = Y - Hk @ W
            sc = np.sum((Aall.T @ Yres) ** 2, 1) / (np.sum(Aall ** 2, 0) + 1e-9); return np.argsort(-sc)
        res, lamd = {}, {}
        keep = order[:K]; Wc, _ = cv_ridge(H[:, keep], Y, (1e-4,)); res["cut"] = run(SwiGLULayer(G[keep], U[keep], Wc.T))
        # SKIP at equal storage, and SKIPFREE upper bound
        for name, kk, R in (("skip", K // 2, 3 * K // 2), ("skipfree30", K, min(30 * K, Aall.shape[1]))):
            kp = order[:kk]; routes = rank_routes(kp)[:R]
            W, lamd[name] = cv_ridge(np.concatenate([H[:, kp], Aall[:, routes]], 1), Y)
            Dk, C = W[:kk].T, W[kk:]
            box, by_l, origs = {}, {}, {}
            for ri, r in enumerate(routes):
                by_l.setdefault(lab[r][0], []).append((lab[r][1], C[ri]))
            for l, lst in by_l.items():
                js = [j for j, _ in lst]; Gl, Ul, _ = orig_weights(model, l)
                origs[l] = model.model.layers[l].mlp; st = Stash(origs[l], Gl[js], Ul[js], box, l); st.mask0 = True
                model.model.layers[l].mlp = st
            res[name] = run(SkipLayer(G[kp], U[kp], Dk, {l: np.stack([c for _, c in lst], 1) for l, lst in by_l.items()}, box))
            for l, m in origs.items():
                model.model.layers[l].mlp = m
        # FOLD: K kept + R routes folded into upstream write columns, sink-guarded upstream layers
        rk = rank_routes(keep)
        for m in mults:
            R = min(m * K, Aall.shape[1]); routes = rk[:R]; name = f"fold{m}"
            W, lamd[name] = cv_ridge(np.concatenate([H[:, keep], Aall[:, routes]], 1), Y)
            Dk, C = W[:K].T, W[K:]
            newD = {l: orig_weights(model, l)[2].copy() for l in lays}
            for ri, r in enumerate(routes):
                l, j = lab[r]; newD[l][:, j] += C[ri]
            origs = {}
            for l in lays:
                Gl, Ul, Dl = orig_weights(model, l); origs[l] = model.model.layers[l].mlp
                model.model.layers[l].mlp = SinkGuard(SwiGLULayer(Gl, Ul, newD[l]), SwiGLULayer(Gl, Ul, Dl))
            res[name] = run(SwiGLULayer(G[keep], U[keep], Dk))
            for l, mm in origs.items():
                model.model.layers[l].mlp = mm
        row = {k: float((v - base).mean()) for k, v in res.items()}; row["lam"] = lamd
        for k in res:
            if k != "cut":
                row[f"{k}_minus_cut"] = boot(res[k] - res["cut"])
                row[f"{k}_recovered"] = float(1 - row[k] / row["cut"])
        out[f"L{L}_K{K}{a.tag}"] = row
        print(L, K, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, f"[{time.time()-t0:.0f}s]", flush=True)
        json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
