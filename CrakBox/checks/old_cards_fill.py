import json, glob, os, sys, random
import numpy as np
R = sys.argv[1]
scores = json.load(open(f"{R}/review2/new/scores.json"))
key = json.load(open(f"{R}/review2/new/key.json"))
print("key sample", json.dumps(list(key.values())[0])[:400])
real = {r["node"]: r for r in scores if not r["wrong"]}
cards = []
for f in sorted(glob.glob(f"{R}/cards2/*.jsonl")):
    a = os.path.basename(f)[:-6]
    i = 0
    for l in open(f):
        if l.strip():
            try: c = json.loads(l)
            except Exception: continue
            c["_agent"] = a; c["_i"] = i; i += 1; cards.append(c)
n_by = {}
for c in cards: n_by[c["_agent"]] = max(n_by.get(c["_agent"], 0), c["_i"] + 1)
have = {c["node"] for c in cards}
print("cards", len(cards), "agents", n_by, "scored", len(real))
readers = {}
for c in cards:
    for j in c.get("inputs") or []:
        readers.setdefault(j["node"], set()).add(c["node"])
    mp = c.get("main_parent") or {}
    if mp.get("node"): readers.setdefault(mp["node"], set()).add(c["node"])
rows = []
pos = {c["node"]: (c["_agent"], c["_i"]) for c in cards}
for c in cards:
    if c["node"] not in real: continue
    ins = {j["node"] for j in c.get("inputs") or []}
    mp = c.get("main_parent") or {}
    if mp.get("node"): ins.add(mp["node"])
    nb = ins | readers.get(c["node"], set())
    rows.append(dict(node=c["node"], layer=c["node"].split(":")[0], agent=c["_agent"], order=c["_i"] / max(1, n_by[c["_agent"]] - 1),
                     nb_cards=len([x for x in nb if x in have]), readers_cards=len(readers.get(c["node"], set())),
                     sure=c.get("sure", 0) or 0, gap=real[c["node"]]["gap"]))
def perm_corr(x, y, groups, n=5000, seed=0):
    x = np.array(x, float); y = np.array(y, float); g = np.array(groups)
    rs = random.Random(seed)
    def stat(yy):
        out = 0; w = 0
        for k in sorted(set(groups)):
            m = g == k
            if m.sum() > 3 and x[m].std() > 0 and yy[m].std() > 0:
                out += m.sum() * np.corrcoef(x[m], yy[m])[0, 1]; w += m.sum()
        return out / w
    s = stat(y); hi = 0
    for _ in range(n):
        yy = y.copy()
        for k in sorted(set(groups)):
            m = np.where(g == k)[0]; yy[m] = y[rs.sample(list(m), len(m))]
        hi += abs(stat(yy)) >= abs(s)
    return s, hi / n
L = [r["layer"] for r in rows]; G = [r["gap"] for r in rows]
print("n", len(rows), "pass", np.mean([g >= .2 for g in G]))
for name in ["order", "nb_cards", "readers_cards", "sure"]:
    s, p = perm_corr([r[name] for r in rows], G, L)
    print(f"gap vs {name}, inside each layer: r={s:.3f} p={p:.3f}")
    s, p = perm_corr([r[name] for r in rows], G, [r["layer"] + r["agent"] for r in rows])
    print(f"   inside layer x agent: r={s:.3f} p={p:.3f}")
for lay in sorted(set(L)):
    v = [r for r in rows if r["layer"] == lay]
    v.sort(key=lambda r: r["order"]); h = len(v) // 2
    print(lay, len(v), "early half pass", round(np.mean([r["gap"] >= .2 for r in v[:h]]), 2), "late half pass", round(np.mean([r["gap"] >= .2 for r in v[h:]]), 2))
v = sorted(rows, key=lambda r: r["nb_cards"]); 
for lo, hi in [(0, 1), (2, 3), (4, 6), (7, 99)]:
    w = [r for r in rows if lo <= r["nb_cards"] <= hi]
    if w: print(f"neighbours with cards {lo}-{hi}: n={len(w)} pass={np.mean([r['gap'] >= .2 for r in w]):.2f} mean gap={np.mean([r['gap'] for r in w]):.2f}")
# fill
print("distinct nodes", len(have), "stage 4 nodes in network 2048, fill", round(len([n for n in have if n.startswith('s4')]) / 2048, 3))
# parents read: share of named input edges whose other end has a card
tot = hit = 0
for c in cards:
    for j in c.get("inputs") or []:
        tot += 1; hit += j["node"] in have
print("named inputs with a card on the other end", hit, "of", tot)
