"""R2 (NEURON-MATH section 7, L7'): project trained writes on the span of the member columns. Weights only (projection);
images only score. Joint span (one coefficient set for all 368 reader inputs) = the statement; per-conv = diagnostic."""
import numpy as np, torch
from inception import *
from family import Data
from w_fisher import load_build, real_kl, A_cols

m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"])
A = A_cols(m, fam); Aj = torch.cat(A, 0)                    # (368, 10), joint span
sizes = [a.shape[0] for a in A]
def joint(cols):
    C = torch.cat(cols, 0); coef = torch.linalg.lstsq(Aj, C).solution; P = Aj @ coef
    out = (C - P).pow(2).sum(0) / C.pow(2).sum(0)
    return list(torch.split(P, sizes, 0)), out
def perconv(cols):
    return [a @ torch.linalg.lstsq(a, c).solution for a, c in zip(A, cols)]
builds = {}
for tag in ["L-k1-m379-lr3e-2", "L-k1-r-lr3e-2", "L-k3-m-lr3e-2"]:
    fm = load_build(m, fam, tag); cols = fm.weights()[2]; zero = [torch.zeros(c.shape[0], device=DEV) for c in cols]
    pj, outside = joint(cols)
    print(f"{tag}: energy outside joint span per unit {[round(float(o), 3) for o in outside]}", flush=True)
    builds[f"{tag} trained"] = (fm, cols, zero)
    builds[f"{tag} joint-span"] = (fm, pj, zero)
    builds[f"{tag} per-conv"] = (fm, perconv(cols), zero)
K = real_kl(m, fam, D, D.dv, builds)
for tag in ["L-k1-m379-lr3e-2", "L-k1-r-lr3e-2", "L-k3-m-lr3e-2"]:
    t = K[f"{tag} trained"]
    print(f"{tag}: trained {t.mean():.4f} (se {t.std() / np.sqrt(len(t)):.4f})")
    for v in ["joint-span", "per-conv"]:
        d = K[f"{tag} {v}"] - t
        print(f"   {v:10s} {K[f'{tag} {v}'].mean():.4f}   paired {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})")
np.savez("cache/r2_span.npz", **{k.replace(' ', '_'): v for k, v in K.items()})
