"""M3 audit: lift of accepted parents, why the rest fail, and a gallery of random late-stage pairs."""
import numpy as np, os, json
from convnext import *
M = np.load(os.path.join(HF, "imagenet", "m3_matrices.npz"))
H = [384]*3 + [768]*3 + [1536]*9 + [3072]*3; OFF = np.concatenate([[0], np.cumsum(H)])
VS = [(v, s) for v in ("view", "point") for s in ("pos", "neg")]
def best(b):
    rate = M[f"{b}_rate"]; R = []
    for v, s in VS:
        C = M[f"{b}_{v}_{s}_C"].astype(np.float32); br = M[f"{b}_{v}_{s}_br"]
        ok = (C >= .8) & (br[None] > rate[:, None]) & (br[None] <= .5)
        par = np.where(ok.any(1), np.argmin(np.where(ok, br[None], 9), 1), -1)
        R.append((par, np.where(par >= 0, br[np.maximum(par, 0)], 9), np.where(par >= 0, C[np.arange(len(rate)), np.maximum(par, 0)], 0), C, br, v, s))
    return rate, R
out = {}; pairs = []; rng = np.random.default_rng(7)
print("block | parented | lift C/breadth median [p10] | breadth ratio median | lift>=3 | ratio<=10 || unparented buckets: only-broad(>.5) / only-narrower / C<.8")
for b in range(1, 18):
    rate, R = best(b); j = np.argmin([r[1] for r in R], 0); n = np.arange(len(rate))
    pb = np.array([R[k][1][i] for i, k in zip(n, j)]); pc = np.array([R[k][2][i] for i, k in zip(n, j)]); has = pb < 9
    lift = pc[has] / pb[has]; ratio = pb[has] / np.maximum(rate[has], 1e-6)
    # unparented: is there any parent with C>=.8? if so, why excluded
    anyC = np.zeros(len(rate), bool); broadOK = np.zeros(len(rate), bool); narrowOK = np.zeros(len(rate), bool)
    for par, _, _, C, br, v, s in R:
        good = C >= .8; anyC |= good.any(1); broadOK |= (good & (br[None] > .5)).any(1); narrowOK |= (good & (br[None] <= rate[:, None])).any(1)
    u = ~has; ob = u & broadOK; on = u & ~broadOK & narrowOK; lc = u & ~anyC
    o = {"parented": float(has.mean()), "lift_med": float(np.median(lift)), "lift_p10": float(np.quantile(lift, .1)), "ratio_med": float(np.median(ratio)),
         "lift>=3": float((lift >= 3).mean()), "ratio<=10": float((ratio <= 10).mean()), "parented_lift3": float(has.mean() * (lift >= 3).mean()),
         "un_only_broad": float(ob.mean()), "un_only_narrower": float(on.mean()), "un_C_below": float(lc.mean())}
    out[b] = o
    print(f"{b:5d} | {o['parented']:.2f} | {o['lift_med']:.1f} [{o['lift_p10']:.1f}] | {o['ratio_med']:.0f} | {o['lift>=3']:.2f} | {o['ratio<=10']:.2f} || {o['un_only_broad']:.2f} / {o['un_only_narrower']:.2f} / {o['un_C_below']:.2f}")
    if b >= 12:
        for i in rng.choice(np.where(has)[0], 1, replace=False):
            k = j[i]; g = R[k][0][i]; pbk = int(np.searchsorted(OFF, g, side="right") - 1)
            pairs.append((b, int(i), pbk, int(g - OFF[pbk]), R[k][5], R[k][6], float(pb[i]), float(rate[i]), float(pc[i])))
tot = sum(out[b]["parented"] * H[b] for b in out) / sum(H[1:]); tot3 = sum(out[b]["parented_lift3"] * H[b] for b in out) / sum(H[1:])
print(f"whole network: parented {tot:.3f}; parented with lift >= 3: {tot3:.3f}")
json.dump(out, open(os.path.join(RES, "vision-m3-audit.json"), "w"), indent=1)
from gallery import gallery
rows = []
for b, n, pbk, pn, v, s, br, rt, c in pairs:
    rows += [(b, n, f"child, on-rate {rt:.3f}"), (pbk, pn, f"   ^ parent ({v}, {s} pole), breadth {br:.2f}, contains {c:.2f}, lift {c/br:.1f}")]
print(gallery(rows, "parented_pairs_late.png", "Random accepted child-parent pairs, blocks 12-17 (child row, then its parent)"))
