"""Dense overlap over all spots of 600 analysis images (target grid; candidate averaged onto it).
For each target/candidate: on-rate (z>0) and strong-rate (z>30% max) of each; P(cand on | target strong);
P(target strong | cand strong); P(cand strongly NEG (<30% of |min|) | target strong); Spearman-ish corr."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from early_lib import *
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "vision2"))
import gallery as G
H = os.path.dirname(os.path.abspath(__file__)); DEEP = os.path.join(H, "..", "..", "deep")
C = json.load(open(os.path.join(H, "..", "..", "s8", "candidates.json")))
TG = [(4, 187), (2, 332), (3, 103), (5, 373), (1, 61)]
NAT = lambda x: json.load(open(os.path.join(DEEP, f"b{x[0]}_n{x[1]}.json")))
G._init(); m = model(); rng = np.random.default_rng(1)
sel = rng.choice(len(G._idx), 600, replace=False)
X = np.stack([np.asarray(G._X[G._idx[i]]) for i in sel])
ALL = []
for t in TG:
    for x in [t] + [tuple(v) for v in C[f"{t[0]}_{t[1]}"].values()]:
        if x not in ALL: ALL.append(x)
Z = multi(ALL, X); out = {}
def pool(z, R):                      # average-pool candidate map onto the target grid
    f = z.shape[1] // R
    return z.reshape(z.shape[0], R, f, R, f).mean((2, 4)) if f > 1 else z
for t in TG:
    zt = Z[t]; R = zt.shape[1]; nt = NAT(t); ts = zt > 0.3 * nt["max"]; tn = zt < 0.3 * nt["min"]
    print(f"\n=== target {t}: on-rate {np.mean(zt > 0):.2f}  strong-rate {ts.mean():.3f}  strong-neg-rate {tn.mean():.3f}")
    print("        cand      on   strong | P(c on|T+) P(c strong|T+) P(c NEG|T+) P(T+|c strong) | P(c on|T-) P(c NEG|T-) | corr")
    out[str(t)] = {}
    for k, v in C[f"{t[0]}_{t[1]}"].items():
        c = tuple(v); nc = NAT(c); zc = pool(Z[c], R) if Z[c].shape[1] >= R else None
        if zc is None: print(k, c, "coarser grid; skip"); continue
        cs = zc > 0.3 * nc["max"]; cn = zc < 0.3 * nc["min"]
        row = dict(on=float((zc > 0).mean()), strong=float(cs.mean()), p_on=float((zc[ts] > 0).mean()), p_strong=float(cs[ts].mean()),
                   p_neg=float(cn[ts].mean()), p_rev=float(ts[cs].mean()) if cs.any() else 0.0, n_on=float((zc[tn] > 0).mean()), n_neg=float(cn[tn].mean()),
                   corr=float(np.corrcoef(zt.ravel(), zc.ravel())[0, 1]), negrate=float(cn.mean()), p_rev_neg=float(ts[cn].mean()) if cn.any() else 0.0)
        out[str(t)][k] = row
        print(f"  {k} {str(c):9s} {row['on']:.2f}  {row['strong']:.3f} |   {row['p_on']:.2f}       {row['p_strong']:.2f}         {row['p_neg']:.2f}        {row['p_rev']:.2f}     |   {row['n_on']:.2f}     {row['n_neg']:.2f}   | {row['corr']:+.2f} | negrate {row['negrate']:.3f} P(T+|c NEG) {row['p_rev_neg']:.3f}")
json.dump(out, open(os.path.join(H, "overlap.json"), "w"), indent=1)
