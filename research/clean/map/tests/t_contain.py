"""Containment check: at each target's top-50 spots (and its 50 most negative spots), what do the candidate
parents / readers do at the same image location? Compared with 300 random spots. Values: median z / parent's max
over the 3000-image set, and the fraction of spots where z > 0 (parent on)."""
import sys, os, json, numpy as np
from early_lib import *
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vision2"))
import gallery as G

FAM = {(1, 351): [(0, 245), (0, 193), (0, 22), (3, 754), (4, 731)],
       (3, 623): [(0, 346), (0, 364), (2, 99), (2, 61), (0, 200), (1, 309), (4, 401), (4, 467), (4, 395)],
       (4, 499): [(1, 365), (1, 136), (0, 32), (3, 16), (1, 309), (3, 169), (5, 375), (5, 302), (5, 189)],
       (5, 135): [(4, 212), (4, 27), (3, 315), (2, 173), (4, 320), (3, 660), (8, 106), (7, 60), (6, 808)]}
G._init(); want = {}
for t, ps in FAM.items():
    for b, n in [t] + ps: want.setdefault(b, []).append(n)
S = G.scan(want); m = G._m; res = {}
RES = lambda b: 56 // 2 ** m.stage_of[b]
rng = np.random.default_rng(0)
for t, ps in FAM.items():
    vm, pm, vn, pn = S[t]; top = np.argsort(-vm)[:50]; neg = np.argsort(vn)[:50]; rnd = rng.choice(len(vm), 300)
    sets = {"top": [(i, pm[i]) for i in top], "neg": [(i, pn[i]) for i in neg], "rand": [(i, rng.integers(RES(t[0]) ** 2)) for i in rnd]}
    imgs_all = sorted(set(i for v in sets.values() for i, _ in v)); pos = {i: k for k, i in enumerate(imgs_all)}
    Z = multi(ps, np.stack([np.asarray(G._X[G._idx[i]]) for i in imgs_all]))
    print(f"\ntarget {t}: parent value at target spots (median z / parent max | frac on)  top / neg / random")
    res[str(t)] = {}
    for p in ps:
        pmax = S[p][0].max(); Rt, Rp = RES(t[0]), RES(p[0]); f = Rp / Rt; row = {}
        for k, v in sets.items():
            vals = []
            for i, q in v:
                r, c = divmod(int(q), Rt); r0, c0 = int(r * f), int(c * f); w = max(int(f), 1)
                vals.append(Z[p][pos[i], r0:r0 + w, c0:c0 + w].mean())
            vals = np.array(vals); row[k] = (float(np.median(vals) / pmax), float((vals > 0).mean()))
        res[str(t)][str(p)] = row
        print(f"  {str(p):10s}", "  ".join(f"{row[k][0]:+.2f} | {row[k][1]:.2f}" for k in ("top", "neg", "rand")))
json.dump(res, open(__file__.replace(".py", ".json"), "w"), indent=1)
