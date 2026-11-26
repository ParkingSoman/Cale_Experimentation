"""N10 step 1: logits of B and T on held and test photos (Base's are in the gap cache). Reproduces N8's KLs first (P1)."""
import copy, os, sys, json, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "gap"))
import common as C
from train_ref import load_mlp_state
OUT = os.path.join(HERE, "..", "results", "read")
def nets():
    base = C.load("base"); cal = C.Cache("cal")
    calib = C.calibrate(base, (x for x, _ in cal.batches(25)))
    TB = torch.load(os.path.join(C.OUT, "T_build.pt"), weights_only=False)
    TT = torch.load(os.path.join(C.OUT, "T.pt"), weights_only=False)
    mB = copy.deepcopy(base); Ss = C.build(mB, calib, .5, "value"); load_mlp_state(mB, TB["state"])
    mT = copy.deepcopy(mB); load_mlp_state(mT, TT["state"])
    return base, mB, mT, Ss, calib
def logits(m, cache, bs=50):
    out = []
    with torch.no_grad():
        for x, _ in cache.batches(bs): out.append(C.tail(m, x).float().cpu())
    return torch.cat(out).numpy()
if __name__ == "__main__":
    t0 = time.time()
    base, mB, mT, Ss, _ = nets()
    res = {}
    for s in ("held", "test"):
        c = C.Cache(s); zB, zT = logits(mB, c), logits(mT, c)
        np.savez(os.path.join(OUT, f"logits_{s}.npz", ), base=c.z.astype(np.float32), B=zB, T=zT, ix=c.ix)
        p = torch.log_softmax(torch.from_numpy(c.z).float(), 1)
        kl = lambda z: float((p.exp() * (p - torch.log_softmax(torch.from_numpy(z), 1))).sum(1).mean())
        res[s] = dict(B=kl(zB), T=kl(zT)); print(s, res[s], f"{time.time()-t0:.0f}s", flush=True)
    torch.save([S.cpu() for S in Ss], os.path.join(OUT, "survivors.pt"))
    json.dump(res, open(os.path.join(OUT, "logits_kl.json"), "w"), indent=1)
