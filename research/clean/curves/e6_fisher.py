"""E6: step0/step1 Fisher write on self-made vs real inputs at equal count (reuses u9_write.pass_)."""
import sys, numpy as np, torch
import u9_write as U
from inception import *
from family import Family
from w_fisher import real_kl
from u4_w3 import lsq_write
import e5_matched as E   # builds fs (self read), fr (real read), Self, ridx, D, m, fam  (runs E5 once more: ~3 min)
N = E.N; m, fam, D = E.m, E.fam, E.D
out = {}
for tag, fm, src, idx in [("self", E.fs, E.Self, np.arange(N)), ("real", E.fr, D, E.ridx)]:
    U.reads = {tag: fm}; U.D = src
    w0 = U.pass_(idx, seed=1); w1 = U.pass_(idx, base=w0, seed=2)
    out[tag + " lsq"] = (fm, *lsq_write(m, fam, src, fm, idx, 3)); out[tag + " step0"] = (fm, *w0[tag]); out[tag + " step1"] = (fm, *w1[tag])
U.D = D
K = real_kl(m, fam, D, D.dv, out)
for n, v in K.items(): print(f"  {n:12s} dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})", flush=True)
for w in ["lsq", "step0", "step1"]:
    d = K["self " + w] - K["real " + w]; print(f"  paired self - real, {w}: {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})")
np.savez(f"cache/e6_fisher_{N}.npz", **{n.replace(' ', '_'): v for n, v in K.items()})
