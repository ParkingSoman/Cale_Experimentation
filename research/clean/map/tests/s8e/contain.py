"""S8 early band: blind containment. For each target, at its top-50 and 50 most negative spots (3000-image set),
what does each candidate C1..C9 do at the same location? vs 300 random spots.
Reports median z / cand max (pos) , frac on, and lift = (mean_top - mean_rand)/std_rand."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from early_lib import *
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "vision2"))
import gallery as G
HERE = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(HERE, "..", "..", "s8", "candidates.json")))
TG = [(4, 187), (2, 332), (3, 103), (5, 373), (1, 61)]
G._init(); want = {}
for t in TG:
    for b, n in [t] + [tuple(v) for v in C[f"{t[0]}_{t[1]}"].values()]:
        if n not in want.setdefault(b, []): want[b].append(n)
S = G.scan(want); m = G._m; res = {}
RES = lambda b: 56 // 2 ** m.stage_of[b]
rng = np.random.default_rng(0)
for t in TG:
    cd = {k: tuple(v) for k, v in C[f"{t[0]}_{t[1]}"].items()}; ps = list(cd.values())
    vm, pm, vn, pn = S[t]; top = np.argsort(-vm)[:50]; neg = np.argsort(vn)[:50]; rnd = rng.choice(len(vm), 300)
    sets = {"top": [(i, pm[i]) for i in top], "neg": [(i, pn[i]) for i in neg], "rand": [(i, rng.integers(RES(t[0]) ** 2)) for i in rnd]}
    imgs_all = sorted(set(i for v in sets.values() for i, _ in v)); pos = {i: k for k, i in enumerate(imgs_all)}
    Z = multi(ps + [t], np.stack([np.asarray(G._X[G._idx[i]]) for i in imgs_all]))
    print(f"\ntarget {t} (max {vm.max():.2f}): cand at target spots: med z/candmax | frac on | lift(sd)   top / neg / rand")
    res[str(t)] = {}
    for k, p in cd.items():
        pmax = S[p][0].max(); Rt, Rp = RES(t[0]), RES(p[0]); f = Rp / Rt; row = {}; V = {}
        for kk, v in sets.items():
            vals = []
            for i, q in v:
                r, c = divmod(int(q), Rt); r0, c0 = int(r * f), int(c * f); w = max(int(f), 1)
                vals.append(Z[p][pos[i], r0:r0 + w, c0:c0 + w].mean())
            V[kk] = np.array(vals)
        mu, sd = V["rand"].mean(), V["rand"].std() + 1e-6
        for kk in V: row[kk] = (float(np.median(V[kk]) / pmax), float((V[kk] > 0).mean()), float((V[kk].mean() - mu) / sd))
        res[str(t)][k] = {"bn": p, **row}
        print(f"  {k} {str(p):10s}", "  ".join(f"{row[kk][0]:+.2f} | {row[kk][1]:.2f} | {row[kk][2]:+.2f}" for kk in ("top", "neg", "rand")))
json.dump(res, open(os.path.join(HERE, "contain.json"), "w"), indent=1)
