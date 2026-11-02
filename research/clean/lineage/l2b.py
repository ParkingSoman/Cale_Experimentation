"""L2b: lineage (settle) order vs energy order, single layer. Rules: LINEAGE-LOG.md L2b.  python3 l2b.py L K"""
import os, sys
import numpy as np
import mlx.core as mx
from common import *
from probe import probe
from l2 import GMLP, ls_out

D = os.path.join(HF, "lineage-data")
L, K = int(sys.argv[1]), int(sys.argv[2]); assert L in DEV_LAYERS


def zs(A):
    A = A - A.mean(0); s = A.std(0); s[s < 1e-12] = np.inf; return A / s


ZT = zs(gelu(np.load(os.path.join(D, "main.npz"))[f"Z{L}"].astype(np.float64)))
settle = np.mean([(zs(gelu(np.load(os.path.join(D, f"step{s}.npz"))[f"Z{L}"].astype(np.float64))) * ZT).mean(0)
                  for s in (1000, 2000, 4000, 8000, 16000, 32000, 64000)], 0)
model, tok = load_pythia()
cal = mx.concatenate([blocks(tok, "train", 32), blocks(tok, "code_train", 32, skip=60)], 0)
dev, code = blocks(tok, "test", 40), blocks(tok, "code_train", 40)
Wi, bi, Wo, bo = (a.astype(np.float64) for a in mlp_w(model, L))
X = mlp_in(model, cal, L).astype(np.float64); Hf = gelu(X @ Wi.T + bi); Y = Hf @ Wo.T + bo
energy = (Hf ** 2).mean(0) * (Wo ** 2).sum(0)
rk = lambda v: np.argsort(np.argsort(-v))
orders = {"energy": np.argsort(-energy), "settle": np.argsort(-settle),
          "hybrid": np.argsort(rk(energy) + rk(settle)), "random": np.random.default_rng(0).permutation(3072)}
base = {"prose": block_ce(model, dev), "code": block_ce(model, code)}
orig = layers(model)[L].mlp
out = {"layer": L, "K": K, "settle_energy_spearman": float(np.corrcoef(rk(energy), rk(settle))[0, 1]),
       "overlap_settle_energy": len(set(orders["settle"][:K]) & set(orders["energy"][:K])) / K, "arms": {}}
per = {}
for name, o in orders.items():
    keep = np.sort(o[:K]); Wo2, bo2 = ls_out(Hf[:, keep], Y)
    layers(model)[L].mlp = GMLP(Wi[keep], bi[keep], Wo2, bo2)
    per[name] = {t: block_ce(model, b) - base[t] for t, b in (("prose", dev), ("code", code))}
    out["arms"][name] = {t: boot(v) for t, v in per[name].items()}; out["arms"][name]["recite"] = probe(model, tok)["recite"]
    layers(model)[L].mlp = orig
    print(name, out["arms"][name], flush=True)
out["vs_energy"] = {a: {t: boot(per[a][t] - per["energy"][t]) for t in ("prose", "code")} for a in per if a != "energy"}
out["vs_random"] = {a: {t: boot(per[a][t] - per["random"][t]) for t in ("prose", "code")} for a in ("settle", "hybrid")}
print("spearman", out["settle_energy_spearman"], "overlap", out["overlap_settle_energy"])
for a, v in out["vs_energy"].items(): print("vs energy", a, v)
save(f"l2b-L{L}-K{K}", out)
