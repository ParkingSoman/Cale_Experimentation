"""Dev check of method files at given layers (dev text only: test blocks 0-39). Never touches held-out data.

  python3 research/clean/coarse/dev_eval.py --methods methods/recipe_v1.py,methods/cut_ls.py --layers 15,28 --K 48
Appends to results/clean-k-dev.json.
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from contract import load_method, ROOT  # noqa: E402
from tree_lib import load_model, token_blocks  # noqa: E402
from r1 import block_ce  # noqa: E402
from dev_children import children_stats  # noqa: E402
from contract import mlp_io, orig_weights  # noqa: E402


def boot(d, n=2000):
    r = np.random.default_rng(0); m = np.array([d[r.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return [float(d.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", required=True); ap.add_argument("--layers", default="15,28"); ap.add_argument("--K", type=int, default=48); ap.add_argument("--code", action="store_true")
    a = ap.parse_args()
    model, tok = load_model()
    cal = token_blocks(tok, "train", 256); dev = token_blocks(tok, "test", 40)
    if a.code:
        import mlx.core as mx
        txt = open(os.path.join(ROOT, "research", "clean", "data", "code_train.txt")).read()
        ids = np.array(tok.encode(txt[:400000]), dtype=np.int32)
        nb = min(40, len(ids) // 513)
        dev = mx.array(ids[:nb * 513].reshape(nb, 513))
    base = block_ce(model, dev)
    Xd = {}
    path = os.path.join(ROOT, "results", "clean-k-dev.json")
    out = json.load(open(path)) if os.path.exists(path) else {"rows": []}
    paths = [os.path.join(HERE, p) for p in a.methods.split(",")]
    for L in map(int, a.layers.split(",")):
        ce = {}
        for p in paths:
            mod = load_method(p); t = time.time()
            layer = mod.build(model, tok, L, a.K, cal)
            secs = time.time() - t
            orig = model.model.layers[L].mlp; model.model.layers[L].mlp = layer
            ce[mod.METHOD["name"]] = block_ce(model, dev); model.model.layers[L].mlp = orig
            if L not in Xd:
                Xd[L] = mlp_io(model, dev[:20], L)[0]
            G_, U_, _ = orig_weights(model, L)
            row = {"method": mod.METHOD["name"], "layer": L, "K": a.K, "params": layer.n_params(), "secs": round(secs),
                   "dce": float((ce[mod.METHOD["name"]] - base).mean()), "text": "code" if a.code else "prose", **children_stats(layer.units, Xd[L][::2], G_, U_)}
            print(row, flush=True); out["rows"].append(row)
        names = list(ce)
        for nm in names[1:]:
            print(f"  L{L} {names[0]} - {nm}: {boot(ce[names[0]] - ce[nm])}", flush=True)
            out["rows"].append({"layer": L, "diff": f"{names[0]} - {nm}", "boot": boot(ce[names[0]] - ce[nm])})
        json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
