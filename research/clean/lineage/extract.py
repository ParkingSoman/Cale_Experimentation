"""L1 data: for each checkpoint, save MLP weights (all layers) and MLP pre-activations at the dev layers on a fixed
token set. Downloads one checkpoint at a time and deletes it after use (except main)."""
import os, sys, time
import numpy as np
import mlx.core as mx
from common import *

REVS = ["step0"] + [f"step{2**k}" for k in range(10)] + [f"step{s}" for s in (1000, 2000, 4000, 8000, 16000, 32000, 64000)] + ["main"]
OUT = os.path.join(HF, "lineage-data")
os.makedirs(OUT, exist_ok=True)

_, tok = load_pythia()
B = mx.concatenate([blocks(tok, "train", 6, skip=300), blocks(tok, "code_test", 2, skip=50)], 0)  # 8 blocks: 6 prose, 2 code
for rev in REVS:
    f = os.path.join(OUT, f"{rev}.npz")
    if os.path.exists(f):
        continue
    t = time.time()
    m, _ = load_pythia("160m", rev)
    d = {}
    for L in range(12):
        Wi, bi, Wo, bo = mlp_w(m, L)
        d[f"Wi{L}"], d[f"bi{L}"], d[f"Wo{L}"] = Wi.astype(np.float16), bi.astype(np.float16), Wo.astype(np.float16)
    for L in DEV_LAYERS + SEALED_LAYERS:
        X = mlp_in(m, B, L)
        Wi, bi, _, _ = mlp_w(m, L)
        d[f"Z{L}"] = (X @ Wi.T + bi).astype(np.float16)
    ce = block_ce(m, B).mean()
    d["ce"] = np.array(ce)
    np.savez(f, **d)
    del m
    drop("160m", rev)
    print(rev, "ce %.3f" % ce, "%.0fs" % (time.time() - t), flush=True)
