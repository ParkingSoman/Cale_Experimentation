"""UPSTREAM PARENT test prep (dev, L15, toy children). See CHILDREN-LOG 'UPSTREAM PARENT test'.
Writes poles/items_up.md (anonymous parent neurons), poles/key_up.json, poles/judge_in3.md, results/clean-upstream-parents.json."""
import json, os, sys
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from toy_recipe import GROUPS, pole, silu, run_grab, load_model, token_blocks, SEQ, L, ROOT
OUT = os.path.join(HERE, "poles"); rng = np.random.default_rng(3)
N_HOT, N_BASE = 50, 500
model, tok = load_model(); m = model.model
W = {k: np.array(getattr(m.layers[L].mlp, k + "_proj").weight, dtype=np.float64) for k in ("gate", "up")}
w = np.array(m.layers[L].post_attention_layernorm.weight, dtype=np.float64)
Wo = [np.array(m.layers[l].self_attn.o_proj.weight, dtype=np.float64) for l in range(L + 1)]
Dn = [np.array(m.layers[l].mlp.down_proj.weight, dtype=np.float64) for l in range(L)]
E = np.array(m.embed_tokens.weight, dtype=np.float64)
cal = token_blocks(tok, "train", 64); ids = np.array(cal[:, :-1]); flat = ids.reshape(-1)
X = np.zeros((flat.size, 576)); RMS = np.zeros(flat.size)
def p1(i, g):
    x = g[("res", L)].reshape(-1, 576).astype(np.float64); r = np.sqrt((x ** 2).mean(1) + 1e-5)
    X[i * SEQ:i * SEQ + len(x)] = x / r[:, None] * w; RMS[i * SEQ:i * SEQ + len(x)] = r
run_grab(model, cal, p1)
valid = np.arange(flat.size) % SEQ >= 2
kids = [s for g, lst in GROUPS.items() if g != "decoy" for s in lst]
def top(score):
    order = np.argsort(-np.where(valid, score, -np.inf)); seen = Counter(); out = []
    for p in order:
        if seen[flat[p]] < 3: out.append(p); seen[flat[p]] += 1
        if len(out) == N_HOT: break
    return np.array(out)
POS, NEG = {}, {}
for s in kids:
    j, sg = pole(s); g, u = X @ W["gate"][j], X @ W["up"][j]; h = silu(g) * u
    POS[s] = top(np.where((g > 0) & (sg * u > 0), sg * h, -np.inf)); NEG[s] = top(np.where((g > 0) & (sg * u < 0), -sg * h, -np.inf))
base = rng.choice(np.where(valid)[0], N_BASE, replace=False)
sel = np.unique(np.concatenate(list(POS.values()) + list(NEG.values()) + [base])); pos_of = {p: i for i, p in enumerate(sel)}
S = len(sel); res = np.zeros((S, 576)); emb = np.zeros((S, 576)); ctx = np.zeros((L + 1, S, 576)); hid = np.zeros((L, S, 1536), np.float32)
def p2(i, g):
    lo, hi = i * SEQ, (i + 4) * SEQ; js = [k for k, p in enumerate(sel) if lo <= p < hi]
    if not js: return
    loc = sel[js] - lo
    res[js] = g[("res", L)].reshape(-1, 576)[loc]; emb[js] = E[flat[sel[js]]]
    for l in range(L + 1): ctx[l, js] = g[("ctx", l)].reshape(-1, 576)[loc]
    for l in range(L): hid[l, js] = g[("h", l)].reshape(-1, 1536)[loc]
run_grab(model, cal, p2)
labels = ["embed"] + [f"head {l}.{h}" for l in range(L + 1) for h in range(9)] + [f"neuron {l}.{n}" for l in range(L) for n in range(1536)]
def gate_contrib(j, positions):
    js = np.array([pos_of[p] for p in positions]); coef = W["gate"][j] * w / RMS[sel[js], None]
    out = [np.einsum("nd,nd->n", coef, emb[js])[:, None]]
    for l in range(L + 1): out.append(((coef @ Wo[l]).reshape(len(js), 9, 64) * ctx[l, js].reshape(len(js), 9, 64)).sum(-1))
    for l in range(L): out.append(hid[l, js] * (coef @ Dn[l]))
    return np.concatenate(out, 1).mean(0)
shared = {}
for s in kids:
    j, _ = pole(s); b = gate_contrib(j, base)
    rp, rn = gate_contrib(j, POS[s]) - b, gate_contrib(j, NEG[s]) - b
    shared[s] = np.where((rp > 0) & (rn > 0), np.minimum(rp, rn), 0.0)
neuron_cols = np.array([i for i, l in enumerate(labels) if l.startswith("neuron")])
top3 = {s: neuron_cols[np.argsort(-shared[s][neuron_cols])[:3]] for s in kids}
cnt = Counter(int(c) for s in kids for c in top3[s]); hubs = {c for c, v in cnt.items() if v > len(kids) / 3}
head_cols = np.array([i for i, l in enumerate(labels) if l.startswith("head")])
parent = {}
for s in kids:
    cands = [c for c in neuron_cols[np.argsort(-shared[s][neuron_cols])[:20]] if c not in hubs]
    parent[s] = int(cands[0])
# activation check + contexts for each parent neuron (pole = sign of its activation at the child's positions)
def nact(col, positions):
    l, n = map(int, labels[col].split()[1].split(".")); return hid[l, [pos_of[p] for p in positions], n]
need = {labels[parent[s]] for s in kids}
full = {lab: np.zeros(flat.size, np.float32) for lab in need}
def p3(i, g):
    for lab in need:
        l, n = map(int, lab.split()[1].split(".")); hcur = g[("h", l)].reshape(-1, 1536)[:, n]; full[lab][i * SEQ:i * SEQ + len(hcur)] = hcur
run_grab(model, cal, p3)
def cstr(p, before=14, after=3):
    b, t = divmod(int(p), SEQ)
    return (tok.decode(ids[b, max(0, t - before):t].tolist()) + " [[" + tok.decode([int(ids[b, t])]) + "]]" + tok.decode(ids[b, t + 1:t + 1 + after].tolist())).replace("\n", " ")
info, items = {}, {}
for s in kids:
    lab = labels[parent[s]]; a = full[lab]
    zp, zn = [(a[x].mean() - a[valid].mean()) / (a[valid].std() + 1e-9) for x in (POS[s], NEG[s])]
    sg = 1.0 if zp + zn >= 0 else -1.0
    order = np.argsort(-np.where(valid, sg * a, -np.inf)); seen = Counter(); cx = []
    for p in order:
        if seen[flat[p]] < 2: cx.append(cstr(p)); seen[flat[p]] += 1
        if len(cx) == 12: break
    info[s] = {"parent": lab, "pole": "+" if sg > 0 else "-", "z_at_pos": float(zp), "z_at_neg": float(zn),
               "top_heads": [labels[c] for c in head_cols[np.argsort(-shared[s][head_cols])[:3]]]}
    items.setdefault(f"{lab}{info[s]['pole']}", cx)
anon_keys = list(items); rng.shuffle(anon_keys); anon = {f"U{n:03d}": k for n, k in enumerate(anon_keys)}; inv = {v: k for k, v in anon.items()}
with open(os.path.join(OUT, "items_up.md"), "w") as f:
    f.write("# Detectors to describe\nEach item is one detector in an early or middle layer of a language model. Lines are text where it "
            "responds most; [[x]] marks the token. It may respond to the surrounding context, not the token itself.\n")
    for a_ in sorted(anon): f.write(f"\n## {a_}\n" + "\n".join(f"- {c}" for c in items[anon[a_]]) + "\n")
grp = json.load(open(os.path.join(ROOT, "results", "clean-poles-revised.json")))["A_groupings_real"]
pairs = []
for s, gname in grp.items():
    real = inv[f"{info[s]['parent']}{info[s]['pole']}"]
    others = [inv[f"{info[o]['parent']}{info[o]['pole']}"] for o in kids if o != s and inv[f"{info[o]['parent']}{info[o]['pole']}"] != real]
    pairs.append({"child": s, "grouping": gname, "unit": real, "real": True})
    pairs.append({"child": s, "grouping": gname, "unit": str(rng.choice(others)), "real": False})
rng.shuffle(pairs)
for i, p in enumerate(pairs): p["id"] = f"C{i:02d}"
json.dump({"anon": anon, "pairs": pairs, "info": info, "hubs": [labels[c] for c in hubs]}, open(os.path.join(OUT, "key_up.json"), "w"), indent=1)
with open(os.path.join(OUT, "judge_in3.md"), "w") as f:
    f.write("For each item: a GROUPING (a kind of text situation) and a DETECTOR id. Would this detector plausibly fire across that "
            "grouping, i.e. respond to both / all kinds inside it (not just one narrow kind, and not to almost any text)? Answer "
            "yes / no with confidence 1-5. Some items are deliberately mismatched.\n")
    for p in sorted(pairs, key=lambda z: z["id"]): f.write(f"- {p['id']}: grouping = \"{p['grouping']}\", detector = {p['unit']}\n")
json.dump({"info": info, "hubs": [labels[c] for c in hubs]}, open(os.path.join(ROOT, "results", "clean-upstream-parents.json"), "w"), indent=1)
for s in kids: print(s, info[s])
print("hubs", [labels[c] for c in hubs])
