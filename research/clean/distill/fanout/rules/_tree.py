"""Shared wrapper for the whole tree build (distill/PAGE.md, distill/tree/build.py). Photo use: none.
Default generic input: fanout/cache/{clip,r18}_self_n256_k32_s1.pt. Stats and tree are cached per process."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(DIST, "tree"))
import build as TB  # noqa: E402

_C = {}
INFOS = []


def _x(net, X):
    if X is not None:
        return X
    return torch.load(os.path.join(DIST, "fanout", "cache", f"{net}_self_n256_k32_s1.pt"), weights_only=True).float()


def vit(T, f, X=None, loo=None):
    X = _x("clip", X)
    key = ("clip", id(T), X.shape, float(X.flatten()[::99991].sum()))
    if key not in _C:
        st = TB.clip_stats(T, X)
        _C[key] = (st, TB.clip_tree(T, st))
    st, tr = _C[key]
    W = TB.build_clip(T, f, X, loo=loo, st=st, tr=tr)
    INFOS.append(W.pop("_infos"))
    return W


def r18(A, f, X=None, loo=None):
    X = _x("r18", X)
    key = ("r18", id(A), X.shape, float(X.flatten()[::99991].sum()))
    if key not in _C:
        st = TB.r18_stats(A.cpu(), X)
        _C[key] = (st, TB.r18_tree(A.cpu(), st))
    st, tr = _C[key]
    return TB.build_r18(A, f, X, loo=loo, st=st, tr=tr)
