"""N9 step 0: the build with ONE block shrunk (the rest is Base). Held KL per block and keep fraction."""
import copy, json, os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "gap"))
import common as C
base = C.load("base"); cal, held = C.Cache("cal"), C.Cache("held")
calib = C.calibrate(base, (x for x, _ in cal.batches(25)))
out = {}
for frac in (0.5, 0.25):
    for g in (0, 4, 14, 25, 28):
        m = copy.deepcopy(base)
        blk = C.blocks(m)[g]; mg, K = calib[g]
        S = C.choose(blk, mg, K, frac, "value"); C.shrink_block(blk, S, mg, K)
        kl = C.evaluate(m, held)[0]["kl"]
        out[f"{frac}_{g}"] = kl
        print(f"keep {frac:.2f} block {g:2d}: held KL {kl:.4f}", flush=True)
os.makedirs(os.path.join(HERE, "..", "results", "one"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "..", "results", "one", "b1_scan.json"), "w"), indent=1)
