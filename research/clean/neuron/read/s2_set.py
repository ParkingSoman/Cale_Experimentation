"""N10 step 2c: the set by the fixed rule (top 30 really-patched neurons by B-order gain on choose), the collective test
(scored on score = test birds), and the fallback: the N5 birds tree, both orders, against random same-block sets."""
import sys, os, json, time, copy
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcommon import *
t0 = time.time(); say = lambda *a: print(*a, f"[{time.time()-t0:.0f}s]", flush=True)
base, mB, mT, Ss, calib = nets(); P = photo_sets()
A = json.load(open(os.path.join(OUT, "s2_attr.json")))
pos = [{int(b): k for k, b in enumerate(S.tolist())} for S in Ss]

def b_plus(surv, dele):
    """B with survivors (g, base index) from T, deleted (g, base index) re-added; blocks holding re-added neurons are rebuilt
    from Base with S + extra (writes re-solved), then the listed survivors there get T's numbers.
    Returns (network, reference): the reference is B with the same blocks rebuilt from S alone (the null)."""
    m, ref = copy.deepcopy(mB), copy.deepcopy(mB)
    extra = {}
    for g, j in dele: extra.setdefault(g, []).append(j)
    for g, js in extra.items():
        rebuild_block(ref, base, g, Ss[g], calib)
        rebuild_block(m, base, g, torch.cat([Ss[g], torch.tensor(js)]), calib)
    with torch.no_grad():
        for g, b in surv:
            if g in extra:
                newS = sorted(Ss[g].tolist() + extra[g]); k = newS.index(b); j = pos[g][b]
                L1(m, g).weight[k].copy_(L1(mT, g).weight[j]); L1(m, g).bias[k].copy_(L1(mT, g).bias[j]); L2(m, g).weight[:, k].copy_(L2(mT, g).weight[:, j])
            else:
                set_neurons(m, mT, [(g, pos[g][b])])
    return m, ref

def t_minus(surv):
    m = copy.deepcopy(mT); set_neurons(m, mB, [(g, pos[g][b]) for g, b in surv]); return m

def test(surv, dele, rows):
    kB, kT = rows.kl(mB), rows.kl(mT); gap = (kB - kT).mean()
    m, ref = b_plus(surv, dele); gain = rows.kl(ref) - rows.kl(m)
    cost = rows.kl(t_minus(surv)) - kT if surv else np.zeros_like(kT)
    return dict(gap=gap, a_cost=cost.mean() / gap, a_ci=(np.array(boot_ci(cost)) / gap).tolist(),
                b_gain=gain.mean() / gap, b_ci=(np.array(boot_ci(gain)) / gap).tolist())

out = {}
# ---- the set by the fixed rule
items = [("s", r["g"], r["base_index"], r["gain"]) for r in A["surv"] + A["rand"]] + [("d", r["g"], r["base_index"], r["gain"]) for r in A["dele"]]
items = sorted(set(items), key=lambda t: -t[3])[:30]
surv = [(g, b) for k, g, b, _ in items if k == "s"]; dele = [(g, b) for k, g, b, _ in items if k == "d"]
say(f"set: {len(surv)} survivors, {len(dele)} deleted; blocks {sorted({g for _, g, _, _ in items})}")
out["set"] = dict(items=items)
for nm in ("choose", "score"):
    out["set"][nm] = test(surv, dele, P[nm]); say("set", nm, out["set"][nm])
r = out["set"]["score"]; out["set"]["passes"] = bool(r["a_cost"] >= .5 and r["b_gain"] >= .5)
say("collective test passes:", out["set"]["passes"])
# ---- fallback: the N5 birds tree, both orders, and random same-block sets
T5 = json.load(open(os.path.join(HERE, "..", "results", "tree", "trees2.json")))["birds"]
tree = [(T5["root_block"], r) for r in T5["roots"]] + [(L["block"], i) for L in T5["levels"] for i in L["ids"]]
ts = [(g, b) for g, b in tree if b in pos[g]]; td = [(g, b) for g, b in tree if b not in pos[g]]
say(f"tree: {len(tree)} neurons, {len(ts)} survivors, {len(td)} deleted")
out["tree"] = dict(surv=ts, dele=td)
for nm in ("choose", "score"):
    out["tree"][nm] = test(ts, td, P[nm]); say("tree", nm, out["tree"][nm])
rng = np.random.default_rng(0); out["tree_random"] = []
for seed in range(5):
    rs, rd = [], []
    for g in sorted({g for g, _ in tree}):
        ns = sum(1 for x in ts if x[0] == g); nd = sum(1 for x in td if x[0] == g)
        h = calib[g][1].shape[0]; Sset = set(Ss[g].tolist()); D = [i for i in range(h) if i not in Sset]
        rs += [(g, int(b)) for b in rng.choice(Ss[g].numpy(), ns, replace=False)]; rd += [(g, int(b)) for b in rng.choice(D, nd, replace=False)]
    r = test(rs, rd, P["choose"]); out["tree_random"].append(r); say("random same-block", seed, r)
json.dump(conv(out), open(os.path.join(OUT, "s2_set.json"), "w"), indent=1)
say("done")
