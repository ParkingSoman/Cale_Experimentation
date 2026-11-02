"""REVISED POLES test prep: describe each child's GATE-OPEN REGION (moderate open positions, both value signs, not the top).
Writes poles/items_region.md (anonymous R ids), poles/key_region.json, poles/judge_in2.md."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from contract import orig_weights
from toy_recipe import GROUPS, pole, silu, run_grab, load_model, token_blocks, SEQ, L
OUT = os.path.join(HERE, "poles"); rng = np.random.default_rng(11)
model, tok = load_model(); m = model.model
blocks = token_blocks(tok, "train", 160); ids = np.array(blocks[:, :-1]); flat = ids.reshape(-1)
w = np.array(m.layers[L].post_attention_layernorm.weight, dtype=np.float64); Xs = []
run_grab(model, blocks, lambda i, g: Xs.append((lambda x: x / np.sqrt((x ** 2).mean(1, keepdims=True) + 1e-5) * w)(g[("res", L)].reshape(-1, 576).astype(np.float64))))
X = np.concatenate(Xs); valid = np.arange(len(X)) % SEQ >= 2
G, U, D = orig_weights(model, L)
kids = [s for g, lst in GROUPS.items() if g != "decoy" for s in lst]
def ctx(p, before=14, after=3):
    b, t = divmod(int(p), SEQ)
    return (tok.decode(ids[b, max(0, t - before):t].tolist()) + " [[" + tok.decode([int(ids[b, t])]) + "]]" + tok.decode(ids[b, t + 1:t + 1 + after].tolist())).replace("\n", " ")
items = {}
for s in kids:
    j, sg = pole(s); g, u = X @ G[j], X @ U[j]; gate = silu(g)
    op = valid & (g > 0)
    lo, hi = np.quantile(gate[op], [0.5, 0.9])
    band = op & (gate >= lo) & (gate <= hi)
    pick = []
    for sgn_ in (1, -1):
        cand = np.where(band & (sgn_ * u > 0))[0]
        pick += rng.choice(cand, min(7, len(cand)), replace=False).tolist()
    rng.shuffle(pick)
    items[s] = [ctx(p) for p in pick]
order = list(kids); rng.shuffle(order)
anon = {f"R{n:03d}": s for n, s in enumerate(order)}; inv = {v: k for k, v in anon.items()}
with open(os.path.join(OUT, "items_region.md"), "w") as f:
    f.write("# Region items to describe\nEach item is the REGION where one detector is switched on (not its strongest moments): "
            "random positions where it is moderately on. [[x]] marks the token. Describe what these positions have in common "
            "(the context, what is happening in the text), how broad it is, and whether it is coherent.\n")
    for a in sorted(anon): f.write(f"\n## {a}\n" + "\n".join(f"- {c}" for c in items[anon[a]]) + "\n")
pk = json.load(open(os.path.join(OUT, "key.json"))); pinv = {v: k for k, v in pk["anon"].items()}
pairs, trip = [], []
for s in kids:
    pairs.append({"x": pinv[f"{s}:POS"], "y": pinv[f"{s}:NEG"], "real": True, "child": s})
    o = rng.choice([k for k in kids if k != s]); pairs.append({"x": pinv[f"{s}:POS"], "y": pinv[f"{o}:NEG"], "real": False, "child": s})
    trip.append({"p": pinv[f"{s}:POS"], "n": pinv[f"{s}:NEG"], "g": inv[s], "real": True, "child": s})
    o = rng.choice([k for k in kids if k != s]); trip.append({"p": pinv[f"{s}:POS"], "n": pinv[f"{s}:NEG"], "g": inv[o], "real": False, "child": s})
rng.shuffle(pairs); rng.shuffle(trip)
for i, p in enumerate(pairs): p["id"] = f"A{i:02d}"
for i, t in enumerate(trip): t["id"] = f"B{i:02d}"
json.dump({"anon_region": anon, "pairs": pairs, "triples": trip, "stats": pk["stats"]}, open(os.path.join(OUT, "key_region.json"), "w"), indent=1)
with open(os.path.join(OUT, "judge_in2.md"), "w") as f:
    f.write("Background: in a language model, a detector's positive and negative sides are often NOT opposites. The negative side "
            "often separates out things that an EARLIER stage grouped together with the positive side (e.g. a detector for "
            "'cars' whose negative side is 'dogs' because an earlier stage grouped cars and dogs by their pose). The shared "
            "grouping can be abstract (e.g. 'a token right after a number', 'text about quantities').\n\n"
            "# Part A: pairs (X, Y)\nIs there a plausible EARLIER GROUPING that contains both X and Y as two kinds within it (not "
            "something generic like 'any text' or 'any token')? Answer yes / no, name the grouping, confidence 1-5. Some pairs are "
            "deliberately mismatched.\n")
    for p in sorted(pairs, key=lambda z: z["id"]): f.write(f"- {p['id']}: X = {p['x']}, Y = {p['y']}\n")
    f.write("\n# Part B: triples (P, N, R)\nR describes the REGION where a detector is switched on. Is R a grouping that contains BOTH P "
            "and N (R covers both kinds and related things, and is not so generic it covers almost all text)? Answer yes / no, "
            "confidence 1-5. Some triples are deliberately mismatched.\n")
    for t in sorted(trip, key=lambda z: z["id"]): f.write(f"- {t['id']}: P = {t['p']}, N = {t['n']}, R = {t['g']}\n")
print("ok", len(anon))
