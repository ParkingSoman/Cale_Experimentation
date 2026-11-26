"""N8: where does survivor s sit on GELU on d*'s top photos (Base inputs, B's read = Base's read)? Per position."""
import os, sys, json
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "gap"))
import common as C
G, D = 28, 1581
base = C.load("base"); tr = C.Cache("train")
g0 = torch.load(os.path.join(HERE, "..", "results", "step", "grad0.pt"), weights_only=False)
rows, dp = g0["rows"], g0["dstar_peak"]
top = torch.argsort(-dp)[:200].numpy()
e2 = [r for r in json.load(open(os.path.join(C.OUT, "e2_reads_T.json")))["rows"] if r["block"] == G]
ss = [r["s"] for r in e2 if r["dstar"] == D]
U = {}
h = C.blocks(base)[G].block[3].register_forward_hook(lambda m, i, o: U.__setitem__(0, o))
pre_top, pre_all, at_dpeak = [], [], []
with torch.no_grad():
    for name, rr in (("top", rows[np.sort(top)]), ("all", rows[::10])):
        for i0 in range(0, len(rr), 50):
            x, _ = tr.get(rr[i0:i0 + 50]); C.tail(base, x); o = U[0]
            if name == "top":
                pre_top.append(o[..., ss].reshape(-1, len(ss)).cpu())
                # s's pre at the position where d* peaks
                flat = o.reshape(o.shape[0], -1, o.shape[-1])
                idx = flat[..., D].argmax(1)
                at_dpeak.append(flat[torch.arange(len(idx)), idx][:, ss].cpu())
            else:
                pre_all.append(o[..., ss].reshape(-1, len(ss)).cpu())
h.remove()
pt, pa, ad = torch.cat(pre_top), torch.cat(pre_all), torch.cat(at_dpeak)
gp = lambda x: torch.autograd.functional.vjp(F.gelu, x, torch.ones_like(x))[1]
for nm, P in (("d* top photos, all positions", pt), ("d* top photos, at d*'s peak position", ad), ("all photos (every 10th), all positions", pa)):
    q = P.quantile(torch.tensor([.1, .5, .9]), dim=0).median(1).values
    g = gp(P)
    print(f"{nm}: pre_s quantiles (median over the {len(ss)} owners) {[round(float(v), 2) for v in q]}; "
          f"share in the dip (pre < -0.75) {float((P < -0.75).float().mean()):.3f}; above 0 {float((P > 0).float().mean()):.3f}; "
          f"mean GELU' {float(g.mean()):+.3f}")
