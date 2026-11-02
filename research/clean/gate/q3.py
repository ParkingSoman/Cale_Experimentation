"""Q3: are the removed neurons at wide K just less predictable from the routes? (CHILDREN-LOG Q3). Local only, no dCE.
Target r = summed output of the REMOVED neurons. Predictors: routes (L-1 hiddens + attention head outputs), and
routes + kept hiddens. Train on 128 or 205 blocks per text, held-out = blocks 205-255 of each text (fixed).

  python3 research/clean/gate/q3.py --layers 16,22 --Ks 48,384,768
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from a0 import grams, cv_ridge_g, LAMS, FORBID, ROOT  # noqa: E402
from q1 import setup  # noqa: E402
from tree_lib import load_model  # noqa: E402
from g12 import mixed_cal  # noqa: E402

NB, SEQ1 = 256, 511   # blocks per text, rows per block (position 0 dropped)


def rows(blocks):
    return np.concatenate([np.arange(b * SEQ1, (b + 1) * SEQ1) for b in blocks])


def r2(Yt, P):
    return float(1 - np.sum((Yt - P) ** 2) / np.sum((Yt - Yt.mean(0)) ** 2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="16,22"); ap.add_argument("--Ks", default="48,384,768")
    a = ap.parse_args(); layers = [int(x) for x in a.layers.split(",")]; assert not FORBID & set(layers)
    model, tok = load_model(); cal = mixed_cal(tok, NB)
    te = rows(list(range(205, 256)) + list(range(NB + 205, 2 * NB)))
    trs = {"n128": rows(list(range(128)) + list(range(NB, NB + 128))), "n205": rows(list(range(205)) + list(range(NB, NB + 205)))}
    path = os.path.join(ROOT, "results", "clean-q3.json"); out = json.load(open(path)) if os.path.exists(path) else {}; t0 = time.time()
    for L in layers:
        s = setup(model, L, cal); R = np.concatenate([s["Hp"], s["Z"]], 1); D = s["W"][2].astype(np.float32)
        order = np.argsort(-s["en"])
        for K in [int(k) for k in a.Ks.split(",")]:
            keep, rem = np.sort(order[:K]), np.sort(order[K:])
            r = (s["H"][:, rem] @ D[:, rem].T).astype(np.float32)
            row = {"removed_energy_share": float(np.sum(r[te] ** 2) / np.sum(s["Y"][te] ** 2))}
            for nm, tr in trs.items():
                for feat, A in (("routes", R), ("routes_kept", np.concatenate([R, s["H"][:, keep]], 1))):
                    W, lam = cv_ridge_g(grams(A[tr], r[tr]), LAMS); W = W.astype(np.float32)
                    row[f"{feat}_{nm}"] = {"r2_train": r2(r[tr], A[tr] @ W), "r2_test": r2(r[te], A[te] @ W), "lam": lam}
            out[f"L{L}_K{K}"] = row; json.dump(out, open(path, "w"), indent=1)
            print(f"L{L} K{K} removed energy {row['removed_energy_share']:.2f} | " + " | ".join(
                f"{k}: tr {v['r2_train']:.3f} te {v['r2_test']:.3f}" for k, v in row.items() if isinstance(v, dict)) + f" [{time.time()-t0:.0f}s]", flush=True)
        del s, R


if __name__ == "__main__":
    main()
