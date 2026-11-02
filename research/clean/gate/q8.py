"""Q8: ceiling check. Free trained fit (shortfit3, 2 min) vs the closed-form cut at K48 and K768 (CHILDREN-LOG Q8)."""
import json, os, sys, time
import numpy as np
import mlx.core as mx
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.join(CO, "methods")); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from a0 import FORBID, ROOT  # noqa: E402
from contract import mlp_io, cut_ls, SwiGLULayer  # noqa: E402
import shortfit3  # noqa: E402
from tree_lib import load_model  # noqa: E402
from r1 import block_ce  # noqa: E402
from g12 import dev_sets, mixed_cal  # noqa: E402
from sink import SinkGuard  # noqa: E402
from contract import orig_weights  # noqa: E402

shortfit3.MINUTES = 2.0
model, tok = load_model(); c = mixed_cal(tok, 128); dev, devc = dev_sets(tok)
cal = mx.array(np.stack([np.array(c[i // 2 + (i % 2) * 128]) for i in range(256)]))   # interleave prose / code
texts = (("prose", dev), ("code", devc)); base = {t: block_ce(model, b) for t, b in texts}
KS = [int(k) for k in (sys.argv[1] if len(sys.argv) > 1 else "48,768").split(",")]
path = os.path.join(ROOT, "results", "clean-q8.json"); out = json.load(open(path)) if os.path.exists(path) else {}; t0 = time.time()
for L in (16, 22):
    lay = model.model.layers[L]; o = lay.mlp; orig = SwiGLULayer(*orig_weights(model, L))
    for K in KS:
        X, _ = mlp_io(model, cal, L); G, U, D, _ = cut_ls(model, L, K, X); del X
        lay.mlp = SinkGuard(SwiGLULayer(G, U, D), orig); cut = {t: block_ce(model, b) for t, b in texts}; lay.mlp = o
        st = shortfit3.build(model, tok, L, K, cal); lay.mlp = SinkGuard(st, orig); tr = {t: block_ce(model, b) for t, b in texts}; lay.mlp = o
        row = {}
        for t, _ in texts:
            dc, dt = float((cut[t] - base[t]).mean()), float((tr[t] - base[t]).mean()); row[t] = {"cut": dc, "trained": dt, "share": 1 - dt / dc}
        out[f"L{L}_K{K}"] = row; json.dump(out, open(path, "w"), indent=1)
        print(f"L{L} K{K}: trained share prose {row['prose']['share']:.0%} code {row['code']['share']:.0%} (cut {row['prose']['cut']:.4f}/{row['code']['cut']:.4f}) [{time.time()-t0:.0f}s]", flush=True)
