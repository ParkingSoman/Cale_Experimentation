"""G3: shared judgment gate (I2 + I3) vs keeping the strongest members. Local test per family. See CHILDREN-LOG G3.

  python3 research/clean/gate/g3.py --layers 12,15,26
"""
import argparse, json, os, sys, time
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.join(CO, "methods"))
from contract import mlp_io, orig_weights, ROOT  # noqa: E402
from tree_lib import load_model, token_blocks  # noqa: E402
import recipe_v4  # noqa: E402

FORBID = {6, 9, 21, 24}


def silu(z):
    return z / (1.0 + np.exp(-z))


def labels_for(L, nh=9, n=1536):
    return [("embed", 0, 0)] + [("head", l, i) for l in range(L + 1) for i in range(nh)] + \
           [("neuron", l, j) for l in range(L) for j in range(n)]


def rank_r_fit(Xf, sf, Ff, r):
    Z = Xf * sf[:, None]
    W = Z.T @ Z; W += np.eye(W.shape[0]) * 1e-6 * np.trace(W) / W.shape[0]
    N = np.linalg.solve(W, Z.T @ Ff)
    ev, V = np.linalg.eigh(W); ev = np.clip(ev, 1e-12, None)
    Wh, Wih = (V * np.sqrt(ev)) @ V.T, (V / np.sqrt(ev)) @ V.T
    Uu, s, Vt = np.linalg.svd(Wh @ N, full_matrices=False)
    return Wih @ (Uu[:, :r] * s[:r]) @ Vt[:r]


def gate_err(Xf, Xe, zf, ze, Ff, Fe, r):
    """Best a, b on fit rows (grid), error on eval rows. Sign of a is searched too."""
    sd = zf.std() + 1e-9; best = (np.inf, None)
    for a in (-4, -2, -1, 1, 2, 4):
        for q in (0.5, 0.8, 0.9, 0.97):
            b = -np.quantile(a / sd * zf, q)
            sf = silu(a / sd * zf + b)
            N = rank_r_fit(Xf, sf, Ff, r)
            e = np.sum((Ff - (Xf * sf[:, None]) @ N) ** 2)
            if e < best[0]:
                best = (e, (a / sd, b, N))
    a, b, N = best[1]
    return float(np.sum((Fe - (Xe * silu(a * ze + b)[:, None]) @ N) ** 2) / (np.sum(Fe ** 2) + 1e-12))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="12,15,26"); a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]; assert not FORBID & set(layers)
    model, tok = load_model(); cal = token_blocks(tok, "train", 64)
    rng = np.random.default_rng(0)
    path = os.path.join(ROOT, "results", "clean-g3.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for L in layers:
        t = time.time()
        X, _ = mlp_io(model, cal, L); G, U, D = orig_weights(model, L)
        Hfull = silu(X @ G.T) * (X @ U.T)
        en = np.linalg.norm(Hfull, axis=0) * np.linalg.norm(D, axis=0)
        cand = np.argsort(-en)[:400]
        keys = recipe_v4.build(model, tok, L, 48, cal, keys_for=cand)
        lab = labels_for(L)
        fams = defaultdict(list)
        for nrn, k in keys.items():
            if k >= 0:
                fams[k].append(nrn)
        fams = {k: v for k, v in fams.items() if 2 <= len(v) <= 12}
        print(f"L{L}: {len(fams)} families, sizes {sorted([len(v) for v in fams.values()], reverse=True)[:15]} [{time.time()-t:.0f}s]", flush=True)
        sub = rng.choice(len(X), 10000, replace=False)
        Xs = X[sub]; Xf, Xe = Xs[:5000], Xs[5000:]; Hs = Hfull[sub]
        w = np.array(model.model.layers[L].post_attention_layernorm.weight, dtype=np.float64)
        act_cache = {}
        rows = []
        for k, mem in fams.items():
            _, l, j = lab[k]
            mem = np.array(mem)
            F = Hs[:, mem] @ D[:, mem].T; Ff, Fe = F[:5000], F[5000:]
            E = np.sum(Fe ** 2) + 1e-12
            Dl = np.array(model.model.layers[l].mlp.down_proj.weight, dtype=np.float64)
            def zread(jj):
                d = Dl[:, jj]; return (Xs / w) @ (d / np.linalg.norm(d))
            if l not in act_cache:
                Xl, _ = mlp_io(model, cal, l); Gl, Ul, _ = orig_weights(model, l); act_cache[l] = (Xl[sub], Gl, Ul)
            Xl, Gl, Ul = act_cache[l]
            jact = silu(Xl @ Gl[j]) * (Xl @ Ul[j])
            jr = rng.integers(Gl.shape[0])
            for r in sorted({1, int(np.ceil(len(mem) / 2))}):
                if r >= len(mem): continue
                order = mem[np.argsort(-en[mem])][:r]
                Dk = np.linalg.lstsq(Hs[:5000][:, order], Ff, rcond=None)[0]
                base = np.sum((Fe - Hs[5000:][:, order] @ Dk) ** 2) / E
                zj = zread(j); zn = zread(jr)
                rows.append({"key": [l, j], "size": len(mem), "r": r, "base": float(base),
                             "jread": gate_err(Xf, Xe, zj[:5000], zj[5000:], Ff, Fe, r),
                             "jact": gate_err(Xf, Xe, jact[:5000], jact[5000:], Ff, Fe, r),
                             "null": gate_err(Xf, Xe, zn[:5000], zn[5000:], Ff, Fe, r)})
        r1 = [x for x in rows if x["r"] == 1]
        med = lambda f: float(np.median([f(x) for x in r1])) if r1 else None
        summ = {"n_fam": len(fams), "n_r1": len(r1), "med_base": med(lambda x: x["base"]), "med_jread": med(lambda x: x["jread"]),
                "med_jact": med(lambda x: x["jact"]), "med_null": med(lambda x: x["null"]),
                "med_gain_jread": med(lambda x: x["base"] - x["jread"]), "med_gain_null": med(lambda x: x["base"] - x["null"]),
                "med_gain_jact": med(lambda x: x["base"] - x["jact"]), "share_jread_wins": float(np.mean([x["base"] > x["jread"] for x in r1])) if r1 else None}
        summ["pass"] = bool(r1 and summ["med_gain_jread"] > 0 and summ["med_gain_jread"] > summ["med_gain_null"])
        out[str(L)] = {"summary": summ, "rows": rows, "secs": round(time.time() - t)}
        print(f"L{L} {summ}", flush=True)
        json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
