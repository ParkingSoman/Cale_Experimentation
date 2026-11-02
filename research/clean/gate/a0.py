"""A0: per-layer map of single-layer RCF vs CUT at wide K (CHILDREN-LOG "RCF SCALE + CHANNELS").
Same fit as rcf_v1 (kept L hiddens + L-1 hiddens + L attention head outputs -> original L output, CV ridge), but with
g12's mixed calibration (128 prose + 128 code blocks) and grams computed once per fold half.

  python3 research/clean/gate/a0.py --Ks 768,384 [--layers 1,2,3]
"""
import argparse, json, os, sys, time
import numpy as np
import mlx.core as mx
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from contract import orig_weights, ROOT, SwiGLULayer  # noqa: E402
from tree_lib import load_model  # noqa: E402
from r1 import block_ce  # noqa: E402
from g6 import silu, boot  # noqa: E402
from g10 import Lin  # noqa: E402
from g12 import capture, dev_sets, mixed_cal  # noqa: E402
from sink import SinkGuard  # noqa: E402

FORBID = {6, 9, 21, 24}
LAMS = (1e-3, 1e-2, 1e-1, 3e-1)


def grams(A, Y):
    """Fold halves (first half / second half of rows), as cv_ridge in g7 splits them."""
    nb = A.shape[0] // 2; out = []
    for s in (slice(0, nb), slice(nb, None)):
        a, y = A[s], Y[s]
        out.append(((a.T @ a).astype(np.float64), (a.T @ y).astype(np.float64), float(np.sum(y.astype(np.float64) ** 2))))
    return out


def solve(G, B, lam):
    M = G.copy(); M[np.diag_indices_from(M)] += lam * np.trace(G) / G.shape[0]
    return np.linalg.solve(M, B)


def cv_ridge_g(gr, lams):
    """Same choice as g7.cv_ridge (train on one half, test on the other, both ways), from precomputed grams."""
    (G1, B1, y1), (G2, B2, y2) = gr; best = (np.inf, lams[0])
    for lam in lams:
        e = 0.0
        for (Gt, Bt, _), (Ge, Be, ye) in (((G1, B1, y1), (G2, B2, y2)), ((G2, B2, y2), (G1, B1, y1))):
            W = solve(Gt, Bt, lam); e += ye - 2 * np.sum(W * Be) + np.sum(W * (Ge @ W))
        if e < best[0]:
            best = (e, lam)
    return solve(G1 + G2, B1 + B2, best[1]), best[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--Ks", default="768,384"); ap.add_argument("--layers", default="")
    ap.add_argument("--ncal", type=int, default=128); ap.add_argument("--out", default="clean-a0.json")
    a = ap.parse_args()
    Ks = [int(k) for k in a.Ks.split(",")]
    layers = [int(x) for x in a.layers.split(",")] if a.layers else [l for l in range(1, 30) if l not in FORBID]
    assert not FORBID & set(layers)
    model, tok = load_model(); cal = mixed_cal(tok, a.ncal); dev, devc = dev_sets(tok)
    base = {"prose": block_ce(model, dev), "code": block_ce(model, devc)}
    path = os.path.join(ROOT, "results", a.out); out = json.load(open(path)) if os.path.exists(path) else {}
    t0 = time.time()
    for L in layers:
        lay = model.model.layers[L]; plf = L - 1 >= 1
        mods = {"x": (lay, "mlp"), "z": (lay.self_attn, "o_proj")}
        if plf:
            mods["xp"] = (model.model.layers[L - 1], "mlp")
        cap = capture(model, cal, mods)
        G, U, D = orig_weights(model, L); X = cap["x"]
        H = (silu(X @ G.T.astype(np.float32)) * (X @ U.T.astype(np.float32))).astype(np.float32); del X
        Y = (H @ D.T.astype(np.float32)).astype(np.float32)
        en = np.linalg.norm(H, axis=0) * np.linalg.norm(D, axis=0)
        routes = []
        if plf:
            Gp, Up, Dp = orig_weights(model, L - 1); Xp = cap["xp"]
            routes.append((silu(Xp @ Gp.T.astype(np.float32)) * (Xp @ Up.T.astype(np.float32))).astype(np.float32)); del Xp
        routes.append(cap["z"]); R = np.concatenate(routes, 1); del routes, cap
        Wo = np.array(lay.self_attn.o_proj.weight, dtype=np.float64)
        o_mlp, o_op = lay.mlp, lay.self_attn.o_proj; o_prev = model.model.layers[L - 1].mlp if plf else None
        orig_L = SwiGLULayer(G, U, D)
        for K in Ks:
            keep = np.sort(np.argsort(-en)[:K]); Hk = H[:, keep]
            res, lams = {}, {}
            Wc, lams["cut"] = cv_ridge_g(grams(Hk, Y), (1e-4,))
            lay.mlp = SinkGuard(SwiGLULayer(G[keep], U[keep], Wc.T), orig_L)
            res["cut"] = {t: block_ce(model, b) for t, b in (("prose", dev), ("code", devc))}
            A = np.concatenate([Hk, R], 1); W, lams["rcf"] = cv_ridge_g(grams(A, Y), LAMS); del A
            lay.mlp = SinkGuard(SwiGLULayer(G[keep], U[keep], W[:K].T), orig_L); i = K
            if plf:
                Cp = W[i:i + Gp.shape[0]]; i += Gp.shape[0]
                model.model.layers[L - 1].mlp = SinkGuard(SwiGLULayer(Gp, Up, Dp + Cp.T), SwiGLULayer(Gp, Up, Dp))
            lay.self_attn.o_proj = SinkGuard(Lin(Wo + W[i:].T), Lin(Wo))
            res["rcf"] = {t: block_ce(model, b) for t, b in (("prose", dev), ("code", devc))}
            lay.mlp, lay.self_attn.o_proj = o_mlp, o_op
            if plf:
                model.model.layers[L - 1].mlp = o_prev
            row = {"lam": lams}
            for t in ("prose", "code"):
                dc, dr = float((res["cut"][t] - base[t]).mean()), float((res["rcf"][t] - base[t]).mean())
                row[t] = {"cut": dc, "rcf": dr, "share": 1 - dr / dc, "rcf_minus_cut": boot(res["rcf"][t] - res["cut"][t])}
            out[f"L{L}_K{K}"] = row; json.dump(out, open(path, "w"), indent=1)
            print(f"L{L} K{K} prose cut {row['prose']['cut']:.4f} rcf {row['prose']['rcf']:.4f} ({row['prose']['share']:.0%}) | "
                  f"code cut {row['code']['cut']:.4f} rcf {row['code']['rcf']:.4f} ({row['code']['share']:.0%}) "
                  f"lam {lams['rcf']} [{time.time()-t0:.0f}s]", flush=True)
        del H, Y, R


if __name__ == "__main__":
    main()
