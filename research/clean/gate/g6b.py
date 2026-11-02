"""G6b (exploratory): fold route capture, scaled. Options: R multiples, BACK, routes only among upstream trunks
(top-K per earlier layer), refit iterations after folding (side effects). Dev CE vs cut_ls.

  python3 research/clean/gate/g6b.py --L 15 --K 48 --mults 3,10,30 --back 6 --iters 0,2 [--trunks]
"""
import argparse, json, os, sys, time
import numpy as np
import mlx.core as mx
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.dirname(HERE))
from contract import mlp_io, orig_weights, ROOT, SwiGLULayer  # noqa: E402
from tree_lib import load_model, token_blocks  # noqa: E402
from r1 import block_ce  # noqa: E402
from g6 import silu, boot, ridge_ls, FORBID, Stash, SkipLayer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=15); ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--mults", default="3,10,30"); ap.add_argument("--back", type=int, default=6)
    ap.add_argument("--iters", default="0"); ap.add_argument("--trunks", action="store_true"); ap.add_argument("--tag", default="")
    ap.add_argument("--cv", action="store_true"); ap.add_argument("--skipfree", action="store_true"); ap.add_argument("--mask0", action="store_true"); ap.add_argument("--maxratio", type=float, default=0)
    a = ap.parse_args(); L, K = a.L, a.K; assert L not in FORBID
    model, tok = load_model(); cal = token_blocks(tok, "train", 64); dev = token_blocks(tok, "test", 40)
    base = block_ce(model, dev)
    G, U, D = orig_weights(model, L)
    def layer_data():
        X, _ = mlp_io(model, cal, L); return X
    X = layer_data(); H = silu(X @ G.T) * (X @ U.T); Y = H @ D.T
    en = np.linalg.norm(H, axis=0) * np.linalg.norm(D, axis=0); keep = np.argsort(-en)[:K]
    lays = list(range(max(0, L - a.back), L)); lab, cols = [], []
    def ups_act():
        out = []
        for l in lays:
            Xl, _ = mlp_io(model, cal, l); Gl, Ul, Dl = orig_weights(model, l)
            Al = silu(Xl @ Gl.T) * (Xl @ Ul.T)
            if a.trunks:
                el = np.linalg.norm(Al, axis=0) * np.linalg.norm(Dl, axis=0); js = np.sort(np.argsort(-el)[:K])
            else:
                js = np.arange(Al.shape[1])
            out.append((l, js, Al[:, js].astype(np.float32)))
        return out
    U0 = ups_act()
    Aall = np.concatenate([x[2] for x in U0], 1); lab = [(l, j) for (l, js, _) in U0 for j in js]
    ok = np.ones(Aall.shape[1], bool)
    if a.maxratio > 0:
        ok = np.abs(Aall).max(0) / (Aall.std(0) + 1e-9) <= a.maxratio
        print(f"spike filter keeps {ok.sum()} of {len(ok)}", flush=True)
    def run(layer):
        orig = model.model.layers[L].mlp; model.model.layers[L].mlp = layer
        ce = block_ce(model, dev); model.model.layers[L].mlp = orig; return ce
    Dk0 = ridge_ls(H[:, keep], Y).T
    ce_cut = run(SwiGLULayer(G[keep], U[keep], Dk0))
    path = os.path.join(ROOT, "results", "clean-g6b.json"); out = json.load(open(path)) if os.path.exists(path) else {}
    rows = {"cut": float((ce_cut - base).mean())}
    print(f"L{L} K{K} cut {rows['cut']:.4f} candidates {Aall.shape[1]}", flush=True)
    Yres = Y - H[:, keep] @ Dk0.T
    sc = np.sum((Aall.T @ Yres) ** 2, 1) / (np.sum(Aall ** 2, 0) + 1e-9); sc[~ok] = -np.inf; rank = np.argsort(-sc)
    nb = X.shape[0] // 2   # rows are block-ordered: first half of blocks vs second half
    def cv_lam(A):
        if not a.cv:
            return 1e-4
        best = (np.inf, 1e-4)
        for lam in (1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0):
            e = 0.0
            for tr, te in ((slice(0, nb), slice(nb, None)), (slice(nb, None), slice(0, nb))):
                W = ridge_ls(A[tr], Y[tr], lam); e += np.sum((Y[te] - A[te] @ W) ** 2)
            if e < best[0]:
                best = (e, lam)
        return best[1]
    orig_W = {l: model.model.layers[l].mlp.down_proj.weight for l in lays}
    for m in map(int, a.mults.split(",")):
        R = min(m * K, Aall.shape[1]); routes = rank[:R]
        if a.skipfree:
            Afit = np.concatenate([H[:, keep], Aall[:, routes]], 1); lam = cv_lam(Afit); W = ridge_ls(Afit, Y, lam)
            Dk, C = W[:K].T, W[K:]
            box, by_l, origs = {}, {}, {}
            for ri, r in enumerate(routes):
                by_l.setdefault(lab[r][0], []).append((lab[r][1], C[ri]))
            for l, lst in by_l.items():
                js = [j for j, _ in lst]; Gl, Ul, _ = orig_weights(model, l)
                origs[l] = model.model.layers[l].mlp; model.model.layers[l].mlp = Stash(origs[l], Gl[js], Ul[js], box, l)
                model.model.layers[l].mlp.mask0 = a.mask0
            ce = run(SkipLayer(G[keep], U[keep], Dk, {l: np.stack([c for _, c in lst], 1) for l, lst in by_l.items()}, box))
            for l, mm in origs.items():
                model.model.layers[l].mlp = mm
            key = f"SKIPFREE_R{m}K"; rows[key] = float((ce - base).mean()); rows[key + "_minus_cut"] = boot(ce - ce_cut); rows[key + "_lam"] = lam
            print(f"  {key}: {rows[key]:.4f}  vs cut {rows[key + '_minus_cut']}  lam {lam}", flush=True)
            continue
        for iters in map(int, a.iters.split(",")):
            Xc, Ac = X, Aall
            for it in range(iters + 1):
                Hc = silu(Xc @ G[keep].T) * (Xc @ U[keep].T)
                Afit = np.concatenate([Hc, Ac[:, routes]], 1); lam = cv_lam(Afit); W = ridge_ls(Afit, Y, lam)
                Dk, C = W[:K].T, W[K:]
                for l in lays:
                    model.model.layers[l].mlp.down_proj.weight = orig_W[l]
                Wnp = {l: np.array(orig_W[l]) for l in lays}
                for ri, r in enumerate(routes):
                    l, j = lab[r]; Wnp[l][:, j] += C[ri].astype(np.float32)
                for l in lays:
                    model.model.layers[l].mlp.down_proj.weight = mx.array(Wnp[l])
                if it < iters:   # side effects: recapture layer-L input and route activations under the folded model
                    Xc = layer_data(); Ac = np.concatenate([x[2] for x in ups_act()], 1)
            ce = run(SwiGLULayer(G[keep], U[keep], Dk))
            for l in lays:
                model.model.layers[l].mlp.down_proj.weight = orig_W[l]
            key = f"R{m}K_it{iters}"
            rows[key] = float((ce - base).mean()); rows[key + "_minus_cut"] = boot(ce - ce_cut); rows[key + "_lam"] = lam
            print(f"  {key}: {rows[key]:.4f}  vs cut {rows[key + '_minus_cut']}  lam {lam}", flush=True)
    out[f"L{L}_K{K}_b{a.back}{'_trunks' if a.trunks else ''}{'_cv' if a.cv else ''}{'_mr' + str(int(a.maxratio)) if a.maxratio else ''}{'_skipfree' if a.skipfree else ''}{'_m0' if a.mask0 else ''}{a.tag}"] = rows
    json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
