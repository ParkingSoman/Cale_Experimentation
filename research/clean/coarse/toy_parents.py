"""TOY RECIPE, rule 4: replace the 20 children in L15 by 4 parents and measure the loss (dev).

Variants (each family's children are removed; one unit per family is added):
  remove     no parent (children just removed)
  dominant   the family's dominant child, value + write refit in closed form to the family's summed output
  H1         gate = mean of the children's gates, same closed-form value + write
  recipe     gate = built from the family's shared recipe (toy_recipe.py), same closed-form value + write
Score: KL(original || modified) and CE on dev test blocks 0-39, paired bootstrap 95% over blocks.

  python3 research/clean/coarse/toy_parents.py
Writes results/clean-toy-parents.json.
"""
import json
import os
import sys

import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from toy_recipe import L, GROUPS, ROOT, load_model, token_blocks, SEQ, pole, silu  # noqa: E402
from r2lib import scale_gate  # noqa: E402
from contract import mlp_io  # noqa: E402
from r1 import set_small  # noqa: E402


def block_kl_ce(model, blocks, teacher_logp, batch=4):
    kl, ce = [], []
    for i in range(0, blocks.shape[0], batch):
        b = blocks[i:i + batch]
        lg = model(b[:, :-1]); lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        tl = teacher_logp[i:i + batch]
        kl.append(np.array((mx.exp(tl) * (tl - lg)).sum(-1)[:, 1:].mean(1)))
        ce.append(np.array(-mx.take_along_axis(lg, b[:, 1:, None], axis=-1).squeeze(-1)[:, 1:].mean(1)))
    return np.concatenate(kl), np.concatenate(ce)


def boot(a, b, n=2000, seed=0):
    d = a - b
    r = np.random.default_rng(seed)
    m = np.array([d[r.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return {"mean": float(d.mean()), "lo": float(np.quantile(m, 0.025)), "hi": float(np.quantile(m, 0.975))}


def main():
    rep = json.load(open(os.path.join(ROOT, "results", "clean-toy-recipe.json")))
    model, tok = load_model()
    mlp = model.model.layers[L].mlp
    G = np.array(mlp.gate_proj.weight, dtype=np.float64)
    U = np.array(mlp.up_proj.weight, dtype=np.float64)
    D = np.array(mlp.down_proj.weight, dtype=np.float64)
    cal = token_blocks(tok, "train", 64)
    dev = token_blocks(tok, "test", 40)
    Xc, _ = mlp_io(model, cal, L)
    tl = []
    for i in range(0, dev.shape[0], 4):
        lg = model(dev[i:i + 4, :-1]); tl.append(lg - mx.logsumexp(lg, axis=-1, keepdims=True))
    teacher = mx.concatenate(tl, 0); mx.eval(teacher)

    fams = [g for g in GROUPS if g != "decoy"]
    kids_all = sorted({pole(s)[0] for g in fams for s in GROUPS[g]})
    keep = np.array([n for n in range(G.shape[0]) if n not in set(kids_all)])
    units = {v: [] for v in ("dominant", "H1", "recipe")}
    for g in fams:
        idx = np.array([pole(s)[0] for s in GROUPS[g]])
        Y = (silu(Xc @ G[idx].T) * (Xc @ U[idx].T)) @ D[:, idx].T          # the family's summed output
        dom = pole(rep["groups"][g]["dominant"])[0]
        gates = {"dominant": G[dom], "H1": G[idx].mean(0), "recipe": np.array(rep["groups"][g]["g_p"])}
        for v, gd in gates.items():
            gg, uu, dd = scale_gate(Xc, gd / np.linalg.norm(gd), Y)
            units[v].append((gg, uu, dd))
    # fair baselines (honesty check, 2026-10-20): dominant child kept intact (own gate, value, write), and the same with
    # only its write refit by least squares to the family's summed output
    for g in fams:
        idx = np.array([pole(s)[0] for s in GROUPS[g]])
        Y = (silu(Xc @ G[idx].T) * (Xc @ U[idx].T)) @ D[:, idx].T
        dom = pole(rep["groups"][g]["dominant"])[0]
        units.setdefault("dominant_intact", []).append((G[dom], U[dom], D[:, dom]))
        a = silu(Xc @ G[dom]) * (Xc @ U[dom])
        units.setdefault("dominant_lsD", []).append((G[dom], U[dom], (a @ Y) / (a @ a)))
    out = {"rows": {}, "note": "20 children removed from L15, 4 units added"}
    res = {}
    for v in ("remove", "dominant", "H1", "recipe", "dominant_intact", "dominant_lsD"):
        Gn, Un, Dn = G[keep], U[keep], D[:, keep]
        if v != "remove":
            Gn = np.vstack([Gn] + [u[0][None] for u in units[v]])
            Un = np.vstack([Un] + [u[1][None] for u in units[v]])
            Dn = np.hstack([Dn] + [u[2][:, None] for u in units[v]])
        set_small(model, L, (Gn, Un, Dn))
        res[v] = block_kl_ce(model, dev, teacher)
        set_small(model, L, (G, U, D))
        out["rows"][v] = {"kl": float(res[v][0].mean()), "ce_minus_orig": None}
        print(f"{v:9s} KL {res[v][0].mean():.5f}  CE {res[v][1].mean():.4f}", flush=True)
    for a, b in [("recipe", "dominant"), ("recipe", "H1"), ("recipe", "remove"), ("dominant", "remove"), ("recipe", "dominant_intact"), ("recipe", "dominant_lsD")]:
        out[f"kl_{a}_minus_{b}"] = boot(res[a][0], res[b][0])
        print(f"KL {a} - {b}: {out[f'kl_{a}_minus_{b}']}", flush=True)
    r = out
    out["rule4"] = bool(r["kl_recipe_minus_dominant"]["hi"] < 0 and r["kl_recipe_minus_H1"]["hi"] < 0)
    print("RULE4", "PASS" if out["rule4"] else "fail")
    json.dump(out, open(os.path.join(ROOT, "results", "clean-toy-parents-fair.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
