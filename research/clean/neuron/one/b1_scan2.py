"""N9 step 0b: harder shrinks of one block, and stage 4's 3 blocks together (held KL)."""
import copy, json, os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "gap"))
import common as C
base = C.load("base"); cal, held = C.Cache("cal"), C.Cache("held")
calib = C.calibrate(base, (x for x, _ in cal.batches(25)))
out = {}
def shrink(m, gs, frac):
    for g in gs:
        blk = C.blocks(m)[g]; mg, K = calib[g]
        S = C.choose(blk, mg, K, frac, "value"); C.shrink_block(blk, S, mg, K)
for gs, fracs in (((28,), (0.1, 0.05, 0.02)), ((25,), (0.1, 0.05)), ((4,), (0.1, 0.05)), ((27, 28, 29), (0.5, 0.25))):
    for f in fracs:
        m = copy.deepcopy(base); shrink(m, gs, f)
        kl = C.evaluate(m, held)[0]["kl"]; out[f"{f}_{gs}"] = kl
        print(f"blocks {gs} keep {f:.2f}: held KL {kl:.4f}", flush=True)
json.dump(out, open(os.path.join(HERE, "..", "results", "one", "b1_scan2.json"), "w"), indent=1)
