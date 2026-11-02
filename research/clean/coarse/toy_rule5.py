"""TOY RULE 5 (built on the UPSTREAM PARENT prep) (dev, L15, toy children). See CHILDREN-LOG 'UPSTREAM PARENT test'.
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

# ---------------- TOY RULE 5 ----------------
import mlx.core as mx
from scipy.cluster.hierarchy import linkage, fcluster
sys.path.insert(0, os.path.dirname(HERE))
from r2lib import scale_gate
from r1 import set_small
from contract import mlp_io
from toy_parents import block_kl_ce, boot
rep = json.load(open(os.path.join(ROOT, "results", "clean-toy-recipe.json")))
IDF = os.environ.get("IDF", "0") == "1"
if IDF:
    top50 = np.stack([np.isin(np.arange(len(labels)), np.argsort(-shared[s])[:50]) for s in kids])
    df = np.maximum(top50.mean(0), 1.0 / len(kids))
    idf = np.log(1.0 / df)
    shared = {s: shared[s] * idf for s in kids}
Vs = np.stack([shared[s] / (np.linalg.norm(shared[s]) + 1e-12) for s in kids])
lab4 = fcluster(linkage(Vs, "average", metric="cosine"), 4, "maxclust")
own = {k: [s for s, c in zip(kids, lab4) if c == k] for k in sorted(set(lab4))}
hand = {g: lst for g, lst in GROUPS.items() if g != "decoy"}
print("own families:", own, flush=True)
def upstream_gate(members, n_top=20):
    sh = np.mean([shared[s] / (np.linalg.norm(shared[s]) + 1e-12) for s in members], 0)
    agree = np.mean([(shared[s] > 0) for s in members], 0)
    score = np.where(agree >= 0.5, sh, 0)
    pos = np.concatenate([np.concatenate([POS[s], NEG[s]]) for s in members]); js = np.array([pos_of[p] for p in pos])
    g = np.zeros(576)
    for k in np.argsort(-score)[:n_top]:
        if score[k] <= 0: break
        lb = labels[k]
        if lb == "embed": v = emb[js]
        elif lb.startswith("head"):
            l, h = map(int, lb.split()[1].split(".")); v = ctx[l, js][:, h * 64:(h + 1) * 64] @ Wo[l][:, h * 64:(h + 1) * 64].T
        else:
            l, n = map(int, lb.split()[1].split(".")); v = hid[l, js, n][:, None].astype(np.float64) * Dn[l][:, n][None, :]
        v = (v / RMS[sel[js], None] * w).mean(0); g += score[k] * v / (np.linalg.norm(v) + 1e-12)
    return g
mlp = model.model.layers[L].mlp
G = np.array(mlp.gate_proj.weight, dtype=np.float64); U = np.array(mlp.up_proj.weight, dtype=np.float64); D = np.array(mlp.down_proj.weight, dtype=np.float64)
Xc, _ = mlp_io(model, cal, L)
dev = token_blocks(tok, "test", 40)
tl = []
for i in range(0, dev.shape[0], 4):
    lg = model(dev[i:i + 4, :-1]); tl.append(lg - mx.logsumexp(lg, axis=-1, keepdims=True))
teacher = mx.concatenate(tl, 0); mx.eval(teacher)
kid_n = sorted({pole(s)[0] for s in kids}); keep = np.array([n for n in range(G.shape[0]) if n not in set(kid_n)])
def unit_for(members, gdir):
    idx = np.array([pole(s)[0] for s in members])
    Y = (silu(Xc @ G[idx].T) * (Xc @ U[idx].T)) @ D[:, idx].T
    return scale_gate(Xc, gdir / np.linalg.norm(gdir), Y)
variants = {"remove": []}
variants["dominant_hand"] = [unit_for(m, G[pole(rep["groups"][g]["dominant"])[0]]) for g, m in hand.items()]
variants["recipe_hand"] = [unit_for(m, np.array(rep["groups"][g]["g_p"])) for g, m in hand.items()]
variants["upstream_hand"] = [unit_for(m, upstream_gate(m)) for g, m in hand.items()]
variants["upstream_own"] = [unit_for(m, upstream_gate(m)) for k, m in own.items()]
res = {}
for v, units in variants.items():
    Gn, Un, Dn_ = G[keep], U[keep], D[:, keep]
    if units:
        Gn = np.vstack([Gn] + [u[0][None] for u in units]); Un = np.vstack([Un] + [u[1][None] for u in units]); Dn_ = np.hstack([Dn_] + [u[2][:, None] for u in units])
    set_small(model, L, (Gn, Un, Dn_)); res[v] = block_kl_ce(model, dev, teacher); set_small(model, L, (G, U, D))
    print(f"{v:14s} KL {res[v][0].mean():.5f}", flush=True)
out = {"own_families": {str(k): v for k, v in own.items()}, "kl": {v: float(r[0].mean()) for v, r in res.items()}}
for a_, b_ in [("upstream_own", "dominant_hand"), ("upstream_own", "recipe_hand"), ("upstream_hand", "recipe_hand"), ("upstream_own", "upstream_hand")]:
    out[f"{a_}-{b_}"] = boot(res[a_][0], res[b_][0]); print(a_, "-", b_, out[f"{a_}-{b_}"], flush=True)
out["rule5"] = bool(out["upstream_own-dominant_hand"]["hi"] < 0 and out["upstream_own-recipe_hand"]["hi"] < 0)
print("RULE5", "PASS" if out["rule5"] else "fail")
json.dump(out, open(os.path.join(ROOT, "results", "clean-toy-rule5b.json" if IDF else "clean-toy-rule5.json"), "w"), indent=1)
