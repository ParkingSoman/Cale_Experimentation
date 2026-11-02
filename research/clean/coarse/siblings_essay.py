"""Essay-style siblings (dev): pairs of poles where EACH sits low (below p10, median) at the other's top-1% positions.
Excludes the two poles of one neuron. Null: random pole pairs. Prints contexts for the strongest pairs.
  python3 research/clean/coarse/siblings_essay.py --layer 15
Writes results/clean-siblings-essay-L{L}.json."""
import argparse, json, os, sys
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from contract import mlp_io, orig_weights, ROOT
from tree_lib import load_model, token_blocks, SEQ
from toy_recipe import silu
ap = argparse.ArgumentParser(); ap.add_argument("--layer", type=int, default=15); ap.add_argument("--n", type=int, default=600)
a = ap.parse_args(); L = a.layer
model, tok = load_model(); cal = token_blocks(tok, "train", 64)
ids_b = np.array(cal[:, :-1]); flat = ids_b.reshape(-1); keep = np.where(np.arange(flat.size) % SEQ >= 2)[0]
X, _ = mlp_io(model, cal, L)                   # drops t=0 only
X = X[np.arange(len(X)) % (SEQ - 1) >= 1]      # also drop t=1 (positions 2..511 remain)
G, U, D = orig_weights(model, L); n = G.shape[0]
H = silu(X @ G.T) * (X @ U.T)
Pall = np.concatenate([H, -H], 1)
energy = (np.maximum(Pall, 0) ** 2).sum(0) * np.tile((D ** 2).sum(0), 2)
cand = np.argsort(-energy)[:a.n]
P = Pall[:, cand]
R = np.empty(P.shape, np.float32)
for j in range(P.shape[1]):
    R[np.argsort(P[:, j]), j] = np.linspace(0, 1, len(P), dtype=np.float32)
nh = len(P) // 100
hot = np.argpartition(-P, nh, axis=0)[:nh].T
M = np.stack([np.median(R[hot[b]], axis=0) for b in range(len(cand))])   # M[b, a] = median pct of a at b's top
M = M.T                                                                   # M[a, b]
same = (cand[:, None] % n) == (cand[None, :] % n)
mutual = (M < 0.1) & (M.T < 0.1) & ~same
iu = np.triu_indices(len(cand), 1)
obs = float(mutual[iu].mean())
print(f"L{L}: {int(mutual[iu].sum())} mutual-negative pairs among {len(iu[0])} ({obs:.4f}); "
      f"co-firing pairs (both > p90): {float(((M > 0.9) & (M.T > 0.9) & ~same)[iu].mean()):.4f}", flush=True)
# null: break the shared position axis by circularly shifting each pole's ranks by a random block offset
rng = np.random.default_rng(0)
Rs = np.stack([np.roll(R[:, j], rng.integers(1, len(R))) for j in range(R.shape[1])], 1)
Ms = np.stack([np.median(Rs[hot[b]], axis=0) for b in range(len(cand))]).T
null = float(((Ms < 0.1) & (Ms.T < 0.1) & ~same)[iu].mean())
print(f"  null (positions shifted): {null:.4f}", flush=True)
def ctx(p, before=8):
    b, t = divmod(int(keep[p]), SEQ)
    return (tok.decode(ids_b[b, max(0, t - before):t].tolist()) + " [[" + tok.decode([int(ids_b[b, t])]) + "]]").replace("\n", " ")
def top_ctx(j, k=4):
    order = np.argsort(-P[:, j]); seen = Counter(); out = []
    for p in order:
        tkn = flat[keep[p]]
        if seen[tkn] < 1: out.append(ctx(p)[-60:]); seen[tkn] += 1
        if len(out) == k: break
    return out
pairs = sorted(zip(*np.where(np.triu(mutual, 1))), key=lambda ab: M[ab] + M[ab[::-1]])
deg = mutual.sum(1)
name = lambda j: f"{cand[j] % n}{'+' if cand[j] < n else '-'}"
print(f"  poles with >= 1 mutual sibling: {float((deg > 0).mean()):.2f}; max siblings of one pole {int(deg.max())}", flush=True)
for a_, b_ in pairs[:12]:
    print(f"\n  {name(a_)} vs {name(b_)}  (pct {M[a_, b_]:.2f} / {M[b_, a_]:.2f})")
    for s in top_ctx(a_): print(f"     A: {s}")
    for s in top_ctx(b_): print(f"     B: {s}")
json.dump({"layer": L, "n_poles": a.n, "mutual_share": obs, "null_share": null, "pairs": [[name(x), name(y)] for x, y in pairs[:200]]},
          open(os.path.join(ROOT, "results", f"clean-siblings-essay-L{L}.json"), "w"), indent=1)
