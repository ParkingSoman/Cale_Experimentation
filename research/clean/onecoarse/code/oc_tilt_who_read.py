"""Reader for oc_tilt_who.py (VECMATH 11). Writes results/oc_tilt_who/READ.md from the saved json only."""
import json
import os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results", "oc_tilt_who")
CUTS = [11, 64, 192, 384]
f = lambda x, d=3: f"{x:+.{d}f}" if x < 0 else f"{x:.{d}f}"


def main():
    R = {c: json.load(open(os.path.join(OUT, f"n{c}.json"))) for c in CUTS}
    X = json.load(open(os.path.join(OUT, "cross.json")))
    XM = json.load(open(os.path.join(OUT, "cross_matched.json")))
    EX = json.load(open(os.path.join(OUT, "examples.json")))
    L = ["# VECMATH 11: which kept neurons aim at a cut neuron, and why", ""]

    L += ["## Q1: does j's aim at d rise when d is cut?", "",
          "All pairs: median rise of aim (c1 -> c2), d entering the cut against d kept at both. Null = labels permuted.", "",
          "| Cuts | build | j | entering | rise, entering | rise, staying | difference | null 99th |", "|---|---|---|---|---|---|---|---|"]
    for k, o in X.items():
        for b in ("L", "R4C"):
            x = o[b]
            L.append(f"| {k.replace('_', '->')} | {b} | {o['n_j']} | {o['n_entering']} | {f(x['median_rise_entering'])} | {f(x['median_rise_staying'])} | {f(x['diff'])} | {f(x['null_p99'])} |")
    L += ["", "Top partner only (added after the run): j's best-aimed d at c2, its aim at c1 and c2. Staying = the same pick among other kept neurons, same set size.", "",
          "| Cuts | build | entering partner: c1 -> c2 | staying pick: c1 -> c2 | extra rise |", "|---|---|---|---|---|"]
    for k, x in XM.items():
        c1, c2, b = k.split("_")
        L.append(f"| {c1}->{c2} | {b} | {f(x['entering_c1'])} -> {f(x['entering_c2'])} | {f(x['staying_c1'])} -> {f(x['staying_c2'])} | "
                 f"{f((x['entering_c2'] - x['entering_c1']) - (x['staying_c2'] - x['staying_c1']))} |")
    if "n339_n476" in X["11_64"]["L"]:
        L += ["", f"n339 aim at n476: cut 11 (n476 kept) {X['11_64']['L']['n339_n476'][0]:.3f} -> cut 64 (n476 cut) {X['11_64']['L']['n339_n476'][1]:.3f} (L); "
              f"{X['11_64']['R4C']['n339_n476'][0]:.3f} -> {X['11_64']['R4C']['n339_n476'][1]:.3f} (R4C)"]

    L += ["", "## Q2, Q5 and the rest: who aims (Spearman of best aim with each cause; shuffle 99th)", "",
          "| Cut | build | parent read (s3:69) | write norm | beta_C mass | co-firing with D | self-correlation (L) |", "|---|---|---|---|---|---|---|"]
    for c in CUTS:
        for b in ("L", "R4C"):
            u = R[c][b]["unit"]
            g = lambda k: f"{f(u[k]['rho'])} ({f(u[k]['null_p99'])})" if u.get(k) else ""
            L.append(f"| {c} | {b} | {g('parent_read')} | {g('write_norm')} | {g('beta_mass')} | {g('cofire_maxD')} | {g('self')} |")

    L += ["", "## Q3, Q4 and the rest: which partner (is j's partner the top-ranked cut neuron on each score?)", "",
          "Hit / chance (chance = 1/|D|). Hits = number of kept neurons. Rank pct = median percentile of the partner on that score (1 = top).", "",
          "| Cut | build | shared parent | whole-read similarity | beta_C | co-firing | shared parent within the read-similarity fifth | known answer (planted) |", "|---|---|---|---|---|---|---|---|"]
    for c in CUTS:
        nK = len(R[c]["K"])
        for b in ("L", "R4C"):
            p = R[c][b]["pair"]
            g = lambda k: f"{p[k]['ratio']:.1f}x ({round(p[k]['hit'] * nK)}; pct {p[k]['partner_rank_pct_median']:.2f})"
            w = p["shared_parent_within_read_fifth"]
            L.append(f"| {c} | {b} | {g('shared_parent')} | {g('read_similarity')} | {g('beta_C')} | {g('cofire')} | {w['ratio']:.1f}x | {p['known_answer_planted_hit']:.2f} |")

    L += ["", "## Vecread: A's reads of the strongest and weakest aimers (R4C), share of the group with the concept in its top 5", "",
          "Reference: A's 512 read kernels (vecread README K5b hubs): potholed .12, stained .11, crystalline .10, marbled .10.", "",
          "| Cut | group | best aim (median) | top concepts |", "|---|---|---|---|"]
    for c in CUTS:
        for gname, g in R[c]["groups"].items():
            cnt = Counter(x for up in g["A_up"] for x in up)
            n = len(g["A_up"])
            L.append(f"| {c} | {gname} {n} | {np.median(g['best_aim']):.3f} | " + ", ".join(f"{k} {v / n:.2f}" for k, v in cnt.most_common(6)) + " |")

    L += ["", "## The 5 strongest aimers per cut (R4C)", ""]
    for c in CUTS:
        L += [f"### Cut {c}", "", "| j -> partner | aim | j's A read | j's tilt | partner's A read | data side: j / tilt / partner | shared parent channel | j's read on s3:69 | self (L) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for t in R[c]["top5"]:
            e = EX.get(f"n{c}_j{t['j']}", {})
            ds = " / ".join(", ".join(x[0] for x in e.get(k, [])[:2]) for k in ("A", "t_R4C", "partner_A"))
            L.append(f"| n{t['j']} -> n{t['partner']} | {t['aim']:.2f} | {', '.join(t['A']['up'][:4])} | {', '.join(t['t_R4C']['up'][:4])} | "
                     f"{', '.join(t['partner_A']['up'][:4])} | {ds} | s3:{t['shared_parent_channel']} ({t['shared_parent_value']:.3f}) | {f(t['p69'])} | {t['self_L']:.3f} |")
        L.append("")
    open(os.path.join(OUT, "READ.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
