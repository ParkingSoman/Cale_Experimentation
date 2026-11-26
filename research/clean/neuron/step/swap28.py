"""N8: how much does block 28's read change (where the takeover lives) matter at the end?
T with block 28's reads (W1, b1) reset to B's; B with block 28's reads from T. Held and test KL."""
import copy, os, sys, json
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "gap"))
import common as C
from train_ref import load_mlp_state
base = C.load("base"); cal = C.Cache("cal")
calib = C.calibrate(base, (x for x, _ in cal.batches(25)))
TB = torch.load(os.path.join(C.OUT, "T_build.pt"), weights_only=False)
TT = torch.load(os.path.join(C.OUT, "T.pt"), weights_only=False)
mB = copy.deepcopy(base); C.build(mB, calib, .5, "value"); load_mlp_state(mB, TB["state"])
mT = copy.deepcopy(mB); load_mlp_state(mT, TT["state"])
def setreads(dst, src, g):
    with torch.no_grad():
        db, sb = C.blocks(dst)[g], C.blocks(src)[g]
        db.block[3].weight.copy_(sb.block[3].weight); db.block[3].bias.copy_(sb.block[3].bias)
held, te = C.Cache("held"), C.Cache("test")
out = {}
for name, m in (("B", mB), ("T", mT)):
    out[name] = {c: C.evaluate(m, s)[0]["kl"] for c, s in (("held", held), ("test", te))}
for g in (28, 27, 29):
    m = copy.deepcopy(mT); setreads(m, mB, g)
    out[f"T, block {g} reads from B"] = {c: C.evaluate(m, s)[0]["kl"] for c, s in (("held", held), ("test", te))}
    m = copy.deepcopy(mB); setreads(m, mT, g)
    out[f"B, block {g} reads from T"] = {c: C.evaluate(m, s)[0]["kl"] for c, s in (("held", held), ("test", te))}
for k, v in out.items():
    print(f"{k:28s} held {v['held']:.4f} test {v['test']:.4f}", flush=True)
json.dump(out, open(os.path.join(HERE, "..", "results", "step", "swap28.json"), "w"), indent=1)
