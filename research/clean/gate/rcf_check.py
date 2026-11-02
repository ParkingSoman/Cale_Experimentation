"""Dev check of a method file that may carry `.extra` (installs extras like the keeper must)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from contract import load_method  # noqa: E402
from tree_lib import load_model, token_blocks  # noqa: E402
from r1 import block_ce  # noqa: E402
from g6 import boot  # noqa: E402
meth, L, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
model, tok = load_model(); cal = token_blocks(tok, "train", 256); dev = token_blocks(tok, "test", 40)
base = block_ce(model, dev); ce = {}
for p in (meth, "cut_ls.py"):
    mod = load_method(os.path.join(CO, "methods", p)); layer = mod.build(model, tok, L, K, cal)
    saved = [(o, a, getattr(o, a)) for o, a, _ in getattr(layer, "extra", [])]
    for o, a, m in getattr(layer, "extra", []):
        setattr(o, a, m)
    orig = model.model.layers[L].mlp; model.model.layers[L].mlp = layer
    ce[p] = block_ce(model, dev); model.model.layers[L].mlp = orig
    for o, a, m in saved:
        setattr(o, a, m)
    print(p, "dCE", round(float((ce[p] - base).mean()), 4), "params", layer.n_params(), flush=True)
print("diff", boot(ce[meth] - ce["cut_ls.py"]))
print("restored base", round(float((block_ce(model, dev) - base).mean()), 6))
