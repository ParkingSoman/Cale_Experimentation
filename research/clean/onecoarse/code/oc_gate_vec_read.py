"""Reader for oc_gate_vec.py (VECMATH 10). Writes results/oc_gate_vec/READ.md from the saved json only."""
import json
import os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results", "oc_gate_vec")
CUTS = [11, 64, 192, 384]
NAMED = {11: [5], 64: [339], 192: [430], 384: [430, 76]}
med = lambda x: float(np.median(np.asarray(x, float)))


def f(x, d=3):
    return f"{x:.{d}f}"


def main():
    p0 = json.load(open(os.path.join(OUT, "p0.json")))
    R = {c: json.load(open(os.path.join(OUT, f"n{c}.json"))) for c in CUTS}
    ex = json.load(open(os.path.join(OUT, "examples.json"))) if os.path.exists(os.path.join(OUT, "examples.json")) else {}
    L = [f"# VECMATH 10: the read tilt, read as vectors, and the stream-error rule", "",
         f"P0 (A's 512 kernels through reads()): top z median {p0['top_z_median']:.3f}, leftover median "
         f"{p0['leftover_median']:.3f}, pass {p0['pass']}", ""]

    L += ["## Checks", "", "| Cut | known answer: share above own shuffle 95th (bar .9) | stream error^2 per cell at C | q along a (share removed) |", "|---|---|---|---|"]
    for c in CUTS:
        r = R[c]
        L.append(f"| {c} | {f(r['known_answer']['share_above_own_p95'], 2)} | {f(r['stream_err2_per_cell_C'], 2)} | {f(med(r['q_along_a_share']), 2)} |")

    L += ["", "## P1: is the tilt reproducible (same unit, L against R4C)", "",
          "| Cut | units | same-unit cosine, median | mismatched median | mismatched 99th | share of units above mismatched 99th |", "|---|---|---|---|---|---|"]
    for c in CUTS:
        s = np.array(R[c]["repro"]["same"])
        L.append(f"| {c} | {len(s)} | {f(med(s))} | {f(R[c]['repro']['mismatch_med'])} | {f(R[c]['repro']['mismatch_p99'])} | {f(np.mean(s > R[c]['repro']['mismatch_p99']), 2)} |")

    L += ["", "## Size of the tilt", "", "| Cut | build | alpha median (read metric) | tilt length / A read length, median | bn1 shift change, median abs |", "|---|---|---|---|---|"]
    for c in CUTS:
        for b in ("L", "R4C"):
            r = R[c][b]
            L.append(f"| {c} | {b} | {f(med(r['alpha']))} | {f(med(r['tilt_rel_len']))} | {f(med(np.abs(r['shift_change'])))} |")

    L += ["", "## P3-P5: does the tilt point along the rule q", "",
          "| Cut | build | cos(t, q) median | shuffle median | share of units above own shuffle 95th | neighbour control median | share cos(t,q) > neighbour |",
          "|---|---|---|---|---|---|---|"]
    for c in CUTS:
        for b in ("L", "R4C"):
            r = R[c][b]
            cr, p95, cn = (np.array(r[k]) for k in ("cos_rule", "cos_shuffle_p95", "cos_neighbour"))
            L.append(f"| {c} | {b} | {f(med(cr))} | {f(med(r['cos_shuffle_med']))} | {f(np.mean(cr > p95), 2)} | {f(med(cn))} | {f(np.mean(cr > cn), 2)} |")

    L += ["", "## P6: does the rule open and close the right cells", "",
          "Predicted read = a + |t| x unit(q), A's threshold. Hit = share of training's new-on (lost) cells that the predicted read also opens (closes). Pooled over units.", "",
          "| Cut | build | training new-on / onA | rule: hit new | shuffle: hit new | neighbour: hit new | training lost / onA | rule: hit lost | shuffle: hit lost | neighbour: hit lost | rule opens / training opens |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in CUTS:
        for b in ("L", "R4C"):
            g = {k: np.array(v, float) for k, v in R[c][b]["gates"].items()}
            onA, new, lost = g["onA"].sum(), g["new"].sum(), g["lost"].sum()
            h = lambda p, k: g[f"{p}_hit_{k}"].sum() / max(g[k].sum(), 1)
            L.append(f"| {c} | {b} | {f(new / onA)} | {f(h('rule', 'new'))} | {f(h('shuffle', 'new'))} | {f(h('neighbour', 'new'))} | "
                     f"{f(lost / onA)} | {f(h('rule', 'lost'))} | {f(h('shuffle', 'lost'))} | {f(h('neighbour', 'lost'))} | {f(g['rule_new'].sum() / max(new, 1), 2)} |")

    L += ["", "Added after run 1 (bar unchanged): the same test with the mean shift removed from each predicted tilt (_zm).", "",
          "Hit = share of training's new-on (lost) cells the predicted read also opens (closes). Opens = cells it opens / cells training opens.", "",
          "| Cut | build | mean shift / spread: training tilt, rule | hit new: rule / shuffle / neighbour / random | hit lost: rule / shuffle / neighbour / random | opens: rule / shuffle / neighbour / random |",
          "|---|---|---|---|---|---|"]
    for c in CUTS:
        for b in ("L", "R4C"):
            g = {k: np.array(v, float) for k, v in R[c][b]["gates"].items()}
            if "rule_zm_new" not in g:
                continue
            h = lambda p, k: g[f"{p}_hit_{k}"].sum() / max(g[k].sum(), 1)
            ms = R[c][b]["mean_shift_over_spread"]
            P = ("rule_zm", "shuffle_zm", "neighbour_zm", "random_zm")
            L.append(f"| {c} | {b} | {f(med(ms['tilt']))}, {f(med(ms['rule']))} | " + " / ".join(f(h(p, 'new'), 2) for p in P) + " | "
                     + " / ".join(f(h(p, 'lost'), 2) for p in P) + " | " + " / ".join(f(g[f'{p}_new'].sum() / max(g['new'].sum(), 1), 2) for p in P) + " |")

    L += ["", "## Added run 3: does the tilt point at the deleted units' reads?", "",
          "Per unit: best cosine of the tilt with a read (its a_j part removed), deleted units against other kept units, same set size m.", "",
          "| Cut | build | m | best cos with a deleted read (median) | best cos with a kept read (median) | share of units deleted > kept |", "|---|---|---|---|---|---|"]
    for c in CUTS:
        for b in ("L", "R4C"):
            td = R[c][b].get("toward_D")
            if td:
                L.append(f"| {c} | {b} | {td['m']} | {f(med(td['best_D_med']))} | {f(med(td['best_K_med']))} | {f(td['share_D_gt_K'], 2)} |")

    L += ["", "## P2 and concepts: what the tilt reads as (vecread s3k, whitened pattern, hub z)", "",
          "| Cut | vector | top z median | leftover median |", "|---|---|---|---|"]
    for c in CUTS:
        for k in ("A", "t_L", "t_R4C", "q"):
            rd = R[c]["readings"][k]
            L.append(f"| {c} | {k} | {f(med([x['top_z'] for x in rd]), 2)} | {f(med([x['leftover'] for x in rd]), 2)} |")

    L += ["", "Most common concepts in the top 5 (share of units). A = A's reads (= C's). t = the tilt.", ""]
    for c in CUTS:
        n = len(R[c]["K"])
        L += [f"### Cut {c} ({n} kept units)", "", "| vector | side | top concepts (share of units) |", "|---|---|---|"]
        for k in ("A", "D", "t_L", "t_R4C", "q"):
            if k not in R[c]["readings"]:
                continue
            nk = len(R[c]["readings"][k])
            for side in ("up", "down"):
                cnt = Counter(x for r in R[c]["readings"][k] for x in r[side])
                L.append(f"| {k} | {side} | " + ", ".join(f"{nm} {v / nk:.2f}" for nm, v in cnt.most_common(6)) + " |")
        # concept agreement t vs q, same unit vs shifted unit
        for b in ("L", "R4C"):
            t, q = R[c]["readings"][f"t_{b}"], R[c]["readings"]["q"]
            same = np.mean([len(set(a["up"]) & set(z["up"])) > 0 for a, z in zip(t, q)])
            shift = np.mean([len(set(a["up"]) & set(z["up"])) > 0 for a, z in zip(t, q[1:] + q[:1])])
            L.append(f"| t_{b} vs q | up shared (>= 1 of top 5) | same unit {same:.2f}, other unit {shift:.2f} |")
        L.append("")

    L += ["## Named units", ""]
    for c in CUTS:
        K = R[c]["K"]
        for j in NAMED[c]:
            i = K.index(j)
            L += [f"### Cut {c}, n{j}", "", "| vector | up (top 5) | down (top 5) | top z | data side: top IoU cells (examples) |", "|---|---|---|---|---|"]
            e = ex.get(f"n{c}_j{j}", {})
            for k, ek in (("A", "A"), ("r_L", "rL"), ("t_L", "tL"), ("t_R4C", "tR4C"), ("q", "q")):
                r = R[c]["readings"][k][i]
                exs = ", ".join(f"{x[0]} {x[2]}" for x in e.get(ek, [])[:3])
                L.append(f"| {k} | {', '.join(r['up'])} | {', '.join(r['down'])} | {r['top_z']:.2f} | {exs} |")
            dk = [k for k in ex if k.startswith(f"n{c}_d")]
            for k in dk:
                if any(k == f"n{c}_d{d}" for d in R[c]["D"]):
                    pass
            for d in {11: [464], 64: [476], 192: [114], 384: [501, 374]}[c]:
                e2 = ex.get(f"n{c}_d{d}")
                if e2 and (c != 384 or (j, d) in ((430, 501), (76, 374))):
                    r = e2["read"]
                    L.append(f"| deleted n{d} (A read) | {', '.join(x[0] for x in r['up'][:5])} | {', '.join(x[0] for x in r['down'][:5])} | {r['up'][0][2]:.2f} | "
                             + ", ".join(f"{x[0]} {x[2]}" for x in e2["examples"][:3]) + " |")
            for b in ("L", "R4C"):
                td = R[c][b].get("toward_D")
                if td:
                    L.append(f"| tilt {b}: closest deleted read | n{td['top_D_unit'][i]} (cos {td['top_D_cos'][i]:+.3f}) | | | |")
            L.append(f"| cos(t, q) L / R4C | {R[c]['L']['cos_rule'][i]:+.3f} / {R[c]['R4C']['cos_rule'][i]:+.3f} | shuffle 95th {R[c]['R4C']['cos_shuffle_p95'][i]:+.3f} | | |")
            L.append("")
    open(os.path.join(OUT, "READ.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
