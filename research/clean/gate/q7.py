"""Q7: route-aware choice of kept neurons (CHILDREN-LOG Q7). Keep the neurons the routes CANNOT predict.
Per neuron j: ridge LS of its hidden h_j on the routes (L-1 hiddens + attention outputs), fit on half the rows;
unexplained energy u_j = |h_j - h_j_hat| (held-out half) * |d_j|. Keep top-K by u (vs top-K by energy). Closed form.

  python3 research/clean/gate/q7.py --layers 16,22 --Ks 48,768
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from a0 import grams, cv_ridge_g, solve, LAMS, FORBID, ROOT  # noqa: E402
from q1 import setup, Installer  # noqa: E402
from tree_lib import load_model  # noqa: E402
from r1 import block_ce  # noqa: E402
from g6 import boot  # noqa: E402
from g12 import dev_sets, mixed_cal  # noqa: E402


def unexplained(H, R, D, lam=1e-2):
    """Per-neuron energy the routes cannot predict, measured on held-out rows (train on even blocks of rows)."""
    n = H.shape[0]; tr = np.arange(n) % 2 == 0; te = ~tr
    (G1, B1, _), (G2, B2, _) = grams(R[tr], H[tr])
    W = solve(G1 + G2, B1 + B2, lam).astype(np.float32)
    res = H[te] - R[te] @ W
    return np.linalg.norm(res, axis=0) * np.linalg.norm(D, axis=0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="16,22"); ap.add_argument("--Ks", default="48,768"); ap.add_argument("--ncal", type=int, default=128)
    a = ap.parse_args(); layers = [int(x) for x in a.layers.split(",")]; assert not FORBID & set(layers)
    model, tok = load_model(); cal = mixed_cal(tok, a.ncal); dev, devc = dev_sets(tok)
    texts = (("prose", dev), ("code", devc)); base = {t: block_ce(model, b) for t, b in texts}
    path = os.path.join(ROOT, "results", "clean-q7.json"); out = json.load(open(path)) if os.path.exists(path) else {}; t0 = time.time()
    for L in layers:
        s = setup(model, L, cal); ins = Installer(model, L); R = np.concatenate([s["Hp"], s["Z"]], 1); P = s["Hp"].shape[1]
        u = unexplained(s["H"], R, s["W"][2])
        print(f"L{L}: median predictable share of energy {np.median(1 - u / s['en']):.2f} [{time.time()-t0:.0f}s]", flush=True)
        for K in [int(k) for k in a.Ks.split(",")]:
            sel = {"energy": np.sort(np.argsort(-s["en"])[:K]), "route": np.sort(np.argsort(-u)[:K])}
            res = {}
            for nm, keep in sel.items():
                Hk = s["H"][:, keep]
                Wc, _ = cv_ridge_g(grams(Hk, s["Y"]), (1e-4,))
                A = np.concatenate([Hk, R], 1); W, _ = cv_ridge_g(grams(A, s["Y"]), LAMS); del A
                ins.install(s, keep, Wc.T, None, None, "fold"); res[f"cut_{nm}"] = {t: block_ce(model, b) for t, b in texts}; ins.restore()
                for place in ("fold", "oracle"):
                    ins.install(s, keep, W[:K].T, W[K:K + P], W[K + P:], place); res[f"{place}_{nm}"] = {t: block_ce(model, b) for t, b in texts}; ins.restore()
            overlap = len(set(sel["energy"]) & set(sel["route"])) / K
            row = {"overlap": overlap}
            for t, _ in texts:
                c = float((res["cut_energy"][t] - base[t]).mean()); row[t] = {"cut_energy": c}
                for arm in res:
                    v = float((res[arm][t] - base[t]).mean()); row[t][arm] = {"dce": v, "share": 1 - v / c}
                row[t]["route_minus_energy_fold"] = boot(res["fold_route"][t] - res["fold_energy"][t])
            out[f"L{L}_K{K}"] = row; json.dump(out, open(path, "w"), indent=1)
            for t, _ in texts:
                print(f"L{L} K{K} {t} (overlap {overlap:.2f}): " + " ".join(f"{k} {v['share']:.0%}" for k, v in row[t].items() if isinstance(v, dict))
                      + f" | route-energy fold {np.round(row[t]['route_minus_energy_fold'], 4)} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
