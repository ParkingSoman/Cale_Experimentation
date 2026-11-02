"""Re-check old methods with and without the sink guard (dev CE, blocks 0-39)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from contract import load_method, orig_weights, ROOT, SwiGLULayer  # noqa: E402
from tree_lib import load_model, token_blocks  # noqa: E402
from r1 import block_ce  # noqa: E402
from g6 import boot  # noqa: E402
from sink import SinkGuard  # noqa: E402
methods = sys.argv[1].split(","); layers = [int(x) for x in sys.argv[2].split(",")]; K = int(sys.argv[3])
model, tok = load_model(); cal = token_blocks(tok, "train", 256); dev = token_blocks(tok, "test", 40)
base = block_ce(model, dev); path = os.path.join(ROOT, "results", "clean-sinkcheck.json")
out = json.load(open(path)) if os.path.exists(path) else {}
for L in layers:
    G, U, D = orig_weights(model, L); orig_L = SwiGLULayer(G, U, D); ce = {}
    for p in methods:
        mod = load_method(os.path.join(CO, "methods", p)); layer = mod.build(model, tok, L, K, cal)
        o = model.model.layers[L].mlp
        model.model.layers[L].mlp = layer; ce[mod.METHOD["name"]] = block_ce(model, dev)
        model.model.layers[L].mlp = SinkGuard(layer, orig_L); ce[mod.METHOD["name"] + "+sink"] = block_ce(model, dev)
        model.model.layers[L].mlp = o
    row = {k: float((v - base).mean()) for k, v in ce.items()}
    for k in ce:
        if k != "cut_ls+sink" and k.endswith("+sink"):
            row[k + "_minus_cutsink"] = boot(ce[k] - ce["cut_ls+sink"])
    out[f"L{L}_K{K}"] = {**out.get(f"L{L}_K{K}", {}), **row}
    print(L, K, {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in row.items()}, flush=True)
    json.dump(out, open(path, "w"), indent=1)
