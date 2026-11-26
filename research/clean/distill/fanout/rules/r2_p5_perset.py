"""P5 r2: per-set greedy order. The round-1 p5rank greedy (written value x read gradient, same probe), but ONLY the
named part is masked while its order is measured; every other set stays full. Fixes round-1 risk 3 (one-set rows
used an order measured with all sets masked together).
Part changed: P5 Rank only. Plain cut. Object (picked by): CLIP MLP neurons / heads / stream dims, or ResNet inner /
stream channels (every unit of the part; picked by the greedy score).
Photo use: NONE (weights + generic input X).
"""
import os
import time

import torch

from rules import p5rank as P

SCHED = (1.0, .95, .85, .75, .65, .5)
_M = {}


def order_part(net, model, X, part, seed=1, sched=SCHED, null=False):
    tag = P.input_tag(X)
    path = os.path.join(P.CACHE, f"r2p5_perset_{net}_{part}_{tag}_s{seed}{'_null' if null else ''}_to{sched[-1]:g}.pt")
    if path in _M:
        return _M[path]
    if os.path.exists(path):
        _M[path] = torch.load(path, weights_only=True)
        return _M[path]
    X = (P.default_input() if X is None else X).float()
    if net == "clip":
        import vitgen
        mdl = vitgen.to(model, P.DEV)
        R = P.probe(net, len(X), seed, model["proj"].shape[1])
    else:
        mdl = P.prep_r18(model)
        R = P.probe(net, len(X), seed, model.fc.weight.shape[0])
    sf = P.sizes_fn(net, model)
    full = sf(1.0)
    names = set(P.parts(net, list(full))[part])

    def size_fn(f):
        s = sf(f)
        return {k: (s[k] if k in names else full[k]) for k in s}
    perm = P.make_perm(full, seed) if null else None
    t0 = time.time()
    res = P._greedy(net, mdl, X, R, full, size_fn, sched, perm)
    C = dict(version=1, net=net, part=part, seed=seed, null=null, stop=sched[-1], sched=list(sched), n=len(X),
             photo_use="none", seconds=round(time.time() - t0, 1), **res)
    torch.save(C, path)
    _M[path] = C
    return C


def make(part, which=None, seed=1, null=False):
    which = which or part

    def vit(T, f, X=None):
        return P.build_vit(T, f, order_part("clip", T, X, part, seed, null=null), which)

    def r18(A, f, X=None):
        return P.build_r18(A, f, order_part("r18", A, X, part, seed, null=null), which)
    return vit, r18
