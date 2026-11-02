"""Checks of three essay claims on SmolLM2 (dev, calibration text). Rules fixed before running (CHILDREN-LOG).
1 value broad / gate specific, 2 neurons share values, 3 siblings on each other's negative poles (toy families, L15)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from contract import mlp_io, orig_weights, ROOT
from tree_lib import load_model, token_blocks, SEQ
from toy_recipe import GROUPS, pole, silu
model, tok = load_model(); cal = token_blocks(tok, "train", 64)
ids_all = np.array(cal[:, :-1]).reshape(-1); keep = np.arange(ids_all.size) % SEQ != 0; ids = ids_all[keep]
out = {}
rng = np.random.default_rng(0)
def distinct_top(a, q=0.01):
    k = max(1, int(len(a) * q)); top = np.argpartition(-a, k)[:k]; return len(set(ids[top].tolist())) / k
def pct_at(a, pos):
    s = np.sort(a); return float(np.median(np.searchsorted(s, a[pos]) / len(s)))
for L in (15, 28):
    X, _ = mlp_io(model, cal, L); G, U, D = orig_weights(model, L)
    sub = rng.choice(len(X), 20000, replace=False); Xs = X[sub]; ids_s = ids[sub]
    gp, up = Xs @ G.T, Xs @ U.T
    # 1: breadth (positive pole regime + x +: gate open & value positive; and gate open & value negative)
    wins, nest = [], []
    for j in range(G.shape[0]):
        s = 1.0 if np.mean(np.maximum(up[:, j], 0) * (gp[:, j] > 0)) >= np.mean(np.maximum(-up[:, j], 0) * (gp[:, j] > 0)) else -1.0
        v, g = s * up[:, j], gp[:, j]
        k = 200
        tv, tg = np.argpartition(-v, k)[:k], np.argpartition(-g, k)[:k]
        dv, dg = len(set(ids_s[tv].tolist())), len(set(ids_s[tg].tolist()))
        wins.append(dv > dg)
        nest.append(pct_at(v, tg) > pct_at(g, tv))
    r1 = {"value_more_distinct": float(np.mean(wins)), "nesting_gate_inside_value": float(np.mean(nest))}
    r1["pass"] = bool(r1["value_more_distinct"] >= 0.6 and r1["nesting_gate_inside_value"] >= 0.6)
    # 2: shared values vs shared gates
    def maxcorr(P):
        Z = (P - P.mean(0)) / (P.std(0) + 1e-9); C = np.abs(Z.T @ Z / len(Z)); np.fill_diagonal(C, 0); return C.max(1)
    mv, mg = maxcorr(up), maxcorr(gp)
    r2 = {"share_value_partner_ge_0.6": float(np.mean(mv >= 0.6)), "share_gate_partner_ge_0.6": float(np.mean(mg >= 0.6)),
          "median_max_value_corr": float(np.median(mv)), "median_max_gate_corr": float(np.median(mg))}
    r2["pass"] = bool(r2["share_value_partner_ge_0.6"] >= 0.2 and r2["share_value_partner_ge_0.6"] > r2["share_gate_partner_ge_0.6"])
    out[L] = {"check1": r1, "check2": r2}
    print(f"L{L} check1 {r1}\nL{L} check2 {r2}", flush=True)
    if L == 15:
        H = silu(X @ G.T) * (X @ U.T)
        nh = len(X) // 100
        def neg_frac(a_idx, a_s, b_idx, b_s):
            a, b = a_s * H[:, a_idx], b_s * H[:, b_idx]
            return pct_at(a, np.argpartition(-b, nh)[:nh])
        sib, rnd = [], []
        for g, lst in GROUPS.items():
            if g == "decoy": continue
            ps = [pole(s) for s in lst]
            for i in range(len(ps)):
                for j in range(len(ps)):
                    if i != j:
                        sib.append(neg_frac(ps[i][0], ps[i][1], ps[j][0], ps[j][1]))
                        r = rng.integers(G.shape[0]); rnd.append(neg_frac(ps[i][0], ps[i][1], r, 1.0))
        sib, rnd = np.array(sib), np.array(rnd)
        r3 = {"pairs": len(sib), "sib_below_p25": float(np.mean(sib < 0.25)), "rand_below_p25": float(np.mean(rnd < 0.25)),
              "sib_above_p90": float(np.mean(sib > 0.9)), "rand_above_p90": float(np.mean(rnd > 0.9)),
              "sib_median_pct": float(np.median(sib)), "rand_median_pct": float(np.median(rnd))}
        r3["pass"] = bool(r3["sib_below_p25"] >= 2 * max(r3["rand_below_p25"], 1e-9))
        out[L]["check3"] = r3
        print(f"L15 check3 {r3}", flush=True)
json.dump(out, open(os.path.join(ROOT, "results", "clean-essay-checks.json"), "w"), indent=1)
