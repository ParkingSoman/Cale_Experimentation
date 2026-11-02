"""Diagnose recipe_v1 on dev: same families, swap one part at a time. Dev test blocks 0-39."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from contract import load_method, SwiGLULayer, ls_down, cut_ls, ROOT
from tree_lib import load_model, token_blocks
from r1 import block_ce
mod = load_method(os.path.join(HERE, "methods/recipe_v1.py"))
model, tok = load_model(); cal = token_blocks(tok, "train", 256); dev = token_blocks(tok, "test", 40)
base = block_ce(model, dev); out = {}
def ev(L, Gk, Uk, X, valid, Y):
    Dk = ls_down(X[valid], Gk, Uk, Y[valid]); lay = SwiGLULayer(Gk, Uk, Dk)
    o = model.model.layers[L].mlp; model.model.layers[L].mlp = lay; c = block_ce(model, dev); model.model.layers[L].mlp = o
    return c
for L in (15, 28):
    mod.build(model, tok, L, 48, cal, log=lambda *a: None); d = mod.DEBUG
    X, valid, Y, G, U, D, H = d["X"], d["valid"], d["Y"], d["G"], d["U"], d["D"], d["H"]
    dom = [d["neur"][f[np.argmax(d["energy"][d["cand"][f]])]] for f in d["fams"]]
    res = {}
    res["V1 recipe gate + family-fit value"] = ev(L, np.stack([u[0] for u in d["units"]]), np.stack([u[1] for u in d["units"]]), X, valid, Y)
    v2 = [mod._scale_gate(X[valid], G[j] / np.linalg.norm(G[j]), sum(d["pole_out"](c) for c in f)[valid]) for j, f in zip(dom, d["fams"])]
    res["V2 dominant gate + family-fit value"] = ev(L, np.stack([u[0] for u in v2]), np.stack([u[1] for u in v2]), X, valid, Y)
    res["V3 dominant neuron as-is"] = ev(L, G[dom], U[dom], X, valid, Y)
    Gc, Uc, Dc, idx = cut_ls(model, L, 48, X[valid])
    res["V5 cut_ls"] = ev(L, Gc, Uc, X, valid, Y)
    print(f"L{L}: overlap dominant-set vs cut-set {len(set(dom) & set(idx.tolist()))}/48")
    for k, v in res.items():
        print(f"L{L} {k:38s} dCE {(v - base).mean():+.4f}", flush=True)
    out[L] = {k: float((v - base).mean()) for k, v in res.items()}
json.dump(out, open(os.path.join(ROOT, "results", "clean-k-diag-v1.json"), "w"), indent=1)
