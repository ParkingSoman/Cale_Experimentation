"""POLES test prep (dev, L15, the 20 toy children). For each child: positive pole (gate open, value with the child's
sign), negative pole (gate open, value opposite), gate-only unit (silu(gate), value constant). Writes anonymous items
for a blind describer and a key for grading. Decoys: pairs mixing items of different children.
  python3 research/clean/coarse/toy_poles.py
"""
import json, os, sys
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from contract import orig_weights, ROOT
from toy_recipe import GROUPS, pole, silu, run_grab, load_model, token_blocks, SEQ, L
OUT = os.path.join(HERE, "poles")
rng = np.random.default_rng(7)
model, tok = load_model(); m = model.model
blocks = token_blocks(tok, "train", 160)
ids = np.array(blocks[:, :-1]); flat = ids.reshape(-1)
w = np.array(m.layers[L].post_attention_layernorm.weight, dtype=np.float64)
Xs = []
def fn(i, g):
    x = g[("res", L)].reshape(-1, 576).astype(np.float64); Xs.append(x / np.sqrt((x ** 2).mean(1, keepdims=True) + 1e-5) * w)
run_grab(model, blocks, fn)
X = np.concatenate(Xs); valid = np.arange(len(X)) % SEQ >= 2
G, U, D = orig_weights(model, L)
kids = [s for g, lst in GROUPS.items() if g != "decoy" for s in lst]
def ctx(p, before=14, after=3):
    b, t = divmod(int(p), SEQ)
    return (tok.decode(ids[b, max(0, t - before):t].tolist()) + " [[" + tok.decode([int(ids[b, t])]) + "]]" +
            tok.decode(ids[b, t + 1:t + 1 + after].tolist())).replace("\n", " ")
def top_ctx(score, k=12):
    order = np.argsort(-np.where(valid, score, -np.inf)); seen = Counter(); out = []
    for p in order:
        if seen[flat[p]] < 2:
            out.append(ctx(p)); seen[flat[p]] += 1
        if len(out) == k: break
    return out
items, key, stats = {}, {}, {}
for s in kids:
    j, sg = pole(s)
    g, u = X @ G[j], X @ U[j]
    h = silu(g) * u
    openg = g > 0
    pos = np.where(openg & (sg * u > 0), sg * h, -np.inf)       # + x + regime of this pole
    neg = np.where(openg & (sg * u < 0), -sg * h, -np.inf)      # gate open, value opposite: the negative pole
    gate = silu(g)
    stats[s] = {"gate_open_share": float(openg[valid].mean()),
                "gate_strong_share": float((gate[valid] > 0.5 * np.quantile(gate[valid], 0.999)).mean()),
                "neg_pole_mass_ratio": float(np.maximum(-sg * h, 0)[valid & openg].sum() / (np.maximum(sg * h, 0)[valid & openg].sum() + 1e-9))}
    for kind, sc in (("POS", pos), ("NEG", neg), ("GATE", gate)):
        key[f"{s}:{kind}"] = None
        items[f"{s}:{kind}"] = top_ctx(sc)
ids_anon = list(items); rng.shuffle(ids_anon)
anon = {f"I{n:03d}": k for n, k in enumerate(ids_anon)}
inv = {v: k for k, v in anon.items()}
with open(os.path.join(OUT, "items.md"), "w") as f:
    f.write("# Items to describe\nEach item is one detector. Lines are text where it responds most; [[x]] marks the exact token.\n"
            "The detector reacts at [[x]] but may be about the surrounding context, not the token itself.\n")
    for a in sorted(anon):
        f.write(f"\n## {a}\n" + "\n".join(f"- {c}" for c in items[anon[a]]) + "\n")
# judge (a): pairs; real = (POS, NEG) of one child; decoy = (POS of child, NEG of another child)
pairs = []
for s in kids:
    pairs.append({"x": inv[f"{s}:POS"], "y": inv[f"{s}:NEG"], "real": True, "child": s})
    o = rng.choice([k for k in kids if k != s])
    pairs.append({"x": inv[f"{s}:POS"], "y": inv[f"{o}:NEG"], "real": False, "child": s})
rng.shuffle(pairs)
# judge (b): triples; real = GATE of the same child; decoy = GATE of another child
trip = []
for s in kids:
    trip.append({"p": inv[f"{s}:POS"], "n": inv[f"{s}:NEG"], "g": inv[f"{s}:GATE"], "real": True, "child": s})
    o = rng.choice([k for k in kids if k != s])
    trip.append({"p": inv[f"{s}:POS"], "n": inv[f"{s}:NEG"], "g": inv[f"{o}:GATE"], "real": False, "child": s})
rng.shuffle(trip)
for i, p in enumerate(pairs): p["id"] = f"A{i:02d}"
for i, t in enumerate(trip): t["id"] = f"B{i:02d}"
json.dump({"anon": anon, "pairs": pairs, "triples": trip, "stats": stats}, open(os.path.join(OUT, "key.json"), "w"), indent=1)
with open(os.path.join(OUT, "judge_in.md"), "w") as f:
    f.write("# Part A: pairs\nFor each pair (X, Y): are X and Y related ALTERNATIVES inside one shared category (like 'taxi' vs "
            "'ambulance' inside 'vehicles', or 'V-shape' vs 'O-shape' inside 'shape of the tip')? Answer related / unrelated, "
            "name the shared category if related, and give confidence 1-5.\n")
    for p in sorted(pairs, key=lambda z: z["id"]): f.write(f"- {p['id']}: X = {p['x']}, Y = {p['y']}\n")
    f.write("\n# Part B: triples\nFor each triple (P, N, Gt): is Gt a BROADER category that contains BOTH P and N (Gt fires on both "
            "kinds and on related things)? Answer yes / no, and confidence 1-5. Answer no if Gt is unrelated, or if Gt is so generic "
            "it would fire on almost anything.\n")
    for t in sorted(trip, key=lambda z: z["id"]): f.write(f"- {t['id']}: P = {t['p']}, N = {t['n']}, Gt = {t['g']}\n")
print(json.dumps(stats, indent=0)[:3000])
