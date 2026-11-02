"""G1: gate-sharing groups and the exact shared-gate merge, local test (see CHILDREN-LOG, GATE FAMILIES).

  python3 research/clean/gate/g1.py [--layers 2,8,12,15,18,23,26,28]
Writes results/clean-g1.json. Dev layers only, calibration text only.
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "coarse"))
from contract import mlp_io, orig_weights, ROOT  # noqa: E402
from tree_lib import load_model, token_blocks  # noqa: E402

FORBID = {6, 9, 21, 24}


def silu(z):
    return z / (1.0 + np.exp(-z))


def groups_at(C, thr):
    Dm = 1.0 - C; np.fill_diagonal(Dm, 0.0); Dm = np.clip((Dm + Dm.T) / 2, 0, None)
    lab = fcluster(linkage(squareform(Dm, checks=False), "average"), t=1.0 - thr, criterion="distance")
    gs = [np.where(lab == c)[0] for c in np.unique(lab)]
    return [g for g in gs if len(g) >= 2]


def merge_err(Xf, Xe, S, H, D, g, r):
    """Relative error of the group's summed output on eval rows: shared-gate rank-r merge vs keep-r + LS write."""
    Ff, Fe = H[0][:, g] @ D[:, g].T, H[1][:, g] @ D[:, g].T
    E = np.sum(Fe ** 2) + 1e-12
    rep = g[np.argmax(np.linalg.norm(H[0][:, g], axis=0) * np.linalg.norm(D[:, g], axis=0))]
    sf, se = S[0][:, rep], S[1][:, rep]
    Z = Xf * sf[:, None]
    W = Z.T @ Z; W += np.eye(W.shape[0]) * 1e-6 * np.trace(W) / W.shape[0]
    N = np.linalg.solve(W, Z.T @ Ff)
    ev, V = np.linalg.eigh(W); ev = np.clip(ev, 1e-12, None)
    Wh, Wih = (V * np.sqrt(ev)) @ V.T, (V / np.sqrt(ev)) @ V.T
    Uu, s, Vt = np.linalg.svd(Wh @ N, full_matrices=False)
    Nr = Wih @ (Uu[:, :r] * s[:r]) @ Vt[:r]
    m_err = np.sum((Fe - (Xe * se[:, None]) @ Nr) ** 2) / E
    # baseline: r strongest members kept whole, LS refit of their writes to F
    en = np.linalg.norm(H[0][:, g], axis=0) * np.linalg.norm(D[:, g], axis=0)
    keep = g[np.argsort(-en)[:r]]
    Hf, He = H[0][:, keep], H[1][:, keep]
    Dk = np.linalg.lstsq(Hf, Ff, rcond=None)[0]
    b_err = np.sum((Fe - He @ Dk) ** 2) / E
    return float(m_err), float(b_err)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="2,8,12,15,18,23,26,28"); a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    assert not FORBID & set(layers), "keeper layers"
    model, tok = load_model(); cal = token_blocks(tok, "train", 64)
    rng = np.random.default_rng(0)
    path = os.path.join(ROOT, "results", "clean-g1.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for L in layers:
        t = time.time()
        X, _ = mlp_io(model, cal, L); G, U, D = orig_weights(model, L)
        sub = rng.choice(len(X), 20000, replace=False); X = X[sub]
        Xf, Xe = X[:15000], X[15000:]
        GP = X @ G.T; S_all = silu(GP); H_all = S_all * (X @ U.T)
        S = (S_all[:15000], S_all[15000:]); H = (H_all[:15000], H_all[15000:])
        Z = (S[0] - S[0].mean(0)) / (S[0].std(0) + 1e-9); C = np.abs(Z.T @ Z / len(Z))
        n = G.shape[0]; res = {}
        for thr in (0.6, 0.45):
            gs = groups_at(C, thr)
            cover = sum(len(g) for g in gs) / n
            rows = []
            for g in gs:
                r_half = int(np.ceil(len(g) / 2))
                for r in sorted({1, r_half}):
                    if r >= len(g): continue
                    m, b = merge_err(Xf, Xe, S, H, D, g, r)
                    gr = rng.choice(n, len(g), replace=False)
                    mn, bn = merge_err(Xf, Xe, S, H, D, gr, r)
                    rows.append({"size": len(g), "r": r, "merge": m, "base": b, "null_merge": mn, "null_base": bn,
                                 "open_share": float(np.mean(GP[:, g] > 0))})
            r1 = [x for x in rows if x["r"] == 1]
            gain = np.array([x["base"] - x["merge"] for x in r1]); ngain = np.array([x["null_base"] - x["null_merge"] for x in r1])
            res[str(thr)] = {"n_groups": len(gs), "cover": cover, "sizes": sorted([len(g) for g in gs], reverse=True)[:20],
                             "median_gain_r1": float(np.median(gain)) if len(gain) else None,
                             "median_null_gain_r1": float(np.median(ngain)) if len(ngain) else None,
                             "share_merge_wins_r1": float(np.mean(gain > 0)) if len(gain) else None,
                             "share_null_wins_r1": float(np.mean(ngain > 0)) if len(ngain) else None,
                             "rows": rows}
            print(f"L{L} thr{thr}: groups {len(gs)} cover {cover:.3f} med gain r1 {res[str(thr)]['median_gain_r1']} "
                  f"(null {res[str(thr)]['median_null_gain_r1']}) merge wins {res[str(thr)]['share_merge_wins_r1']}", flush=True)
        out[str(L)] = res; out[str(L)]["secs"] = round(time.time() - t)
        json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
