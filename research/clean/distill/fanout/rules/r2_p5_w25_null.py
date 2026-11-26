"""P5 r2 NULL of r2_p5_w25: same form, MLP activity vector permuted inside each block (stream/heads activity kept)."""
from rules import r2_p5_lib as L


def vit(T, f, X=None):
    W, A = L.scores(T, X)
    import torch
    Ap = dict(A, mlp=[L._perm(v, 100 + i) for i, v in enumerate(A["mlp"])])
    key = "x" if X is not None else "pink"
    old = L._C[key]
    L._C[key] = (W, Ap)
    try:
        return L.build(T, f, X, 1.0, ("stream", "heads", "mlp"), {"mlp": .25})
    finally:
        L._C[key] = old
