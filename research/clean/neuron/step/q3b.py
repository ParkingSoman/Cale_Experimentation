"""N8 Q3b: does s's step-0 read gradient, applied as a read change, move s onto d*'s photos (generalization)?
Base's inputs u at block 28, E2's 3,000 photos, peak over positions. Read changes of equal size (|T - B| of s):
(a) -g0, (b) -sign(g0), (c) T - B. Correlation of each change of s's peak with d*'s Base peak.
Also the same for every E2 survivor of block 28 (with its own d*)."""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gap"))
import common as C  # noqa: E402

G = 28
ap = argparse.ArgumentParser()
ap.add_argument("--grad", default=os.path.join(HERE, "..", "results", "step", "grad0.pt"))
ap.add_argument("--state", default="", help="checkpoint: (a), (b) start from its reads; (c) = its reads - B's")
ap.add_argument("--tag", default="q3b")
ap.add_argument("--cache", default="train", help="photo set: train (E2's 3 per class) or held / test (never trained on)")
A = ap.parse_args()
base = C.load("base")
g0 = torch.load(A.grad, weights_only=False)
TB = torch.load(os.path.join(C.OUT, "T_build.pt"), weights_only=False)
TT = torch.load(os.path.join(C.OUT, "T.pt"), weights_only=False)
rows = g0["rows"] if A.cache == "train" else None
S = torch.as_tensor(TB["S"][G]).numpy()
key = "1.block.3.weight"                               # stage 4 = features[7], block 1 = g 28
WB, WT = TB["state"][2][key].to(C.DEV), TT["state"][2][key].to(C.DEV)
bB = TB["state"][2]["1.block.3.bias"].to(C.DEV)
if A.state:   # (c) becomes the change so far (checkpoint - B); (a), (b) start at the checkpoint, size = |T - checkpoint|
    WT, WB0 = torch.load(A.state, weights_only=False)["state"][2][key].to(C.DEV), WB
    bB = torch.load(A.state, weights_only=False)["state"][2]["1.block.3.bias"].to(C.DEV)
GW1 = g0["acc"]["real"]["W1"].to(C.DEV)
tr = C.Cache(A.cache)
if rows is None:
    rows = np.arange(len(tr))
U = {}
h = C.blocks(base)[G].block[3].register_forward_hook(lambda m, i, o: U.__setitem__(0, (i[0], o)))
e2 = [r for r in json.load(open(os.path.join(C.OUT, "e2_reads_T.json")))["rows"] if r["block"] == G]
ks = [int(np.where(S == r["s"])[0][0]) for r in e2]
ds = [r["dstar"] for r in e2]
P = {n: [] for n in ("base", "start", "a", "b", "c", "e", "d")}
with torch.no_grad():
    dW = {}
    WTend = TT["state"][2][key].to(C.DEV)
    start = WT if A.state else WB
    for j, k in enumerate(ks):
        d = WT[k] - WB[k]
        n = (WTend[k] - start[k]).norm()
        dW[j] = dict(a=-GW1[k] / GW1[k].norm() * n, b=-torch.sign(GW1[k]) / torch.sign(GW1[k]).norm() * n, c=d,
                     e=WTend[k] - start[k])
    for i0 in range(0, len(rows), 50):
        x, _ = tr.get(rows[i0:i0 + 50])
        C.tail(base, x)
        u, pre = U[0]
        P["d"].append(pre[..., ds].amax((1, 2)).cpu())
        pk = []
        for n in ("base", "start", "a", "b", "c", "e"):
            cols = []
            for j, k in enumerate(ks):
                if n == "base":
                    w = WB[k]
                elif n == "c":
                    w = WB[k] + dW[j]["c"]
                else:
                    w = start[k] + (0 if n == "start" else dW[j][n])
                cols.append((u @ w + (TB["state"][2]["1.block.3.bias"].to(C.DEV)[k] if n in ("base", "c") and not A.state else bB[k])).amax((1, 2)))
            P[n].append(torch.stack(cols, 1).cpu())
h.remove()
P = {n: torch.cat(v).double() for n, v in P.items()}
Z = lambda A: (A - A.mean(0)) / A.std(0).clamp_min(1e-12)
out = []
for j, r in enumerate(e2):
    row = dict(s=r["s"], dstar=r["dstar"], e2_owner_corr=r["owner_corr"])
    for n in ("a", "b", "c", "e"):
        dpk = P[n][:, j] - P["base" if n == "c" else "start"][:, j]
        row[n] = float((Z(dpk[:, None])[:, 0] * Z(P["d"][:, j:j + 1])[:, 0]).mean())
    out.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
med = {n: float(np.median([o[n] for o in out])) for n in ("a", "b", "c", "e")}
print("median over the 20 E2 survivors of block 28:", {k: round(v, 3) for k, v in med.items()})
json.dump(dict(rows=out, median=med), open(os.path.join(HERE, "..", "results", "step", f"{A.tag}.json"), "w"), indent=1)
