"""Control for recipe_fit3 (family "baseline-cutfit-frozen"): cut_ls gates FROZEN, same 3-minute value/write fit."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contract import mlp_io, cut_ls, SwiGLULayer  # noqa: E402
from _fit import fit_window  # noqa: E402

METHOD = {"name": "cutfit3_frozen", "family": "baseline-cutfit-frozen", "minutes": 3.5}


def build(model, tok, L, K, cal, log=print):
    X, _ = mlp_io(model, cal[:64], L)
    G, U, D, _ = cut_ls(model, L, K, X)
    layer, steps = fit_window(model, L, SwiGLULayer(G, U, D), cal, minutes=3.0, freeze=("G",))
    return layer
