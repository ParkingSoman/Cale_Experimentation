"""T3: size tree. Children = Pythia-160M neurons (dev layers). Parents = Pythia-70M neurons (all layers) vs a
same-size null (Pythia-160M-seed1, random sample of equal count). Rules: LINEAGE-LOG.md Round 2."""
import numpy as np
from common import *
from cores import *

m, tok = load_pythia("160m"); B = big_blocks(tok); N = B.shape[0] * (SEQ - 1)
child = {L: core(preact(m, B, L), 50) for L in DEV_LAYERS}
del m
par = {}
m70, tok70 = load_pythia("70m")
assert tok70.encode("Hello world, test 123") == tok.encode("Hello world, test 123")
c, lay = [], []
for L in range(len(layers(m70))):
    c.append(core(preact(m70, B, L), 200)); lay += [L] * c[-1].shape[0]
par["70m"] = (np.concatenate(c), np.array(lay)); del m70
ms, _ = load_pythia("160m-seed1")
c, lay = [], []
for L in range(len(layers(ms))):
    c.append(core(preact(ms, B, L), 200)); lay += [L] * c[-1].shape[0]
C, lay = np.concatenate(c), np.array(lay)
pick = np.sort(np.random.default_rng(0).choice(len(C), len(par["70m"][0]), replace=False))
par["seed1"] = (C[pick], lay[pick]); del ms
out = {}
for L in DEV_LAYERS:
    out[L] = {}
    for name, (Cp, lp) in par.items():
        cov = coverage(child[L], Cp, N)
        st, anc, has, ins = split_stats(child[L], cov)
        st["parent_layer_counts"] = np.bincount(lp[anc[has]], minlength=12).tolist()
        out[L][name] = st; print(L, name, st, flush=True)
    a, b = out[L]["70m"]["split_share"], out[L]["seed1"]["split_share"]
    out[L]["pass"] = bool(a >= .10 and a >= 2 * b)
    print(L, "pass", out[L]["pass"])
save("t3", out)
