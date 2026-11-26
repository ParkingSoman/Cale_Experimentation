"""Print board.json as markdown tables, one per network (latest row per rule, f, split, n). Usage: python table.py"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "board.json")))
last = {}
for r in rows:
    last[(r["net"], r["rule"], r["f"], r["split"], r["n"])] = r
for net in ("clip", "r18"):
    print(f"\n### {net}\n")
    print("| Rule | f | Photos | KL to big | Top-1 | Agrees with big | Params | Note |")
    print("|---|---|---|---|---|---|---|---|")
    for k, r in sorted(last.items(), key=lambda kv: (kv[0][1].startswith("check"), kv[0][1], -kv[0][2], kv[0][3], kv[0][4])):
        if k[0] == net:
            print(f"| {r['rule']} | {r['f']:g} | {r['split']} {r['n']} | {r['kl']:.3f} | {100 * r['top1']:.1f}% | "
                  f"{100 * r['agree']:.1f}% | {r['params']:,} | {r.get('note', '')} |")
