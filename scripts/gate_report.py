"""Paired comparison of ladder rungs on the same questions (McNemar-style z).

usage: gate_report.py [results/bench/ladder-gate.json]
Compares ocean, r8 and r4 with each other, and r4 with the qg4 cached-prefill bench
(results/bench/qg4-L.json, level "full") as a harness check.
"""
import json
import math
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/bench/ladder-gate.json"
g = json.load(open(path))
ref = json.load(open("results/bench/qg4-L.json"))["full"]


def paired(a, b):
    """a, b: {id: ok}. Returns n, acc a, acc b, a-only wins, b-only wins, z of (a - b)."""
    ids = sorted(set(a) & set(b))
    wa = sum(1 for i in ids if a[i] and not b[i])
    wb = sum(1 for i in ids if b[i] and not a[i])
    z = (wa - wb) / math.sqrt(wa + wb) if wa + wb else 0.0
    return len(ids), sum(a[i] for i in ids) / len(ids), sum(b[i] for i in ids) / len(ids), wa, wb, z


rows = []
for task in g["ocean"]:
    ok = {s: {it["id"]: it["ok"] for it in g[s][task]["items"]} for s in g}
    q = {it["id"]: it["ok"] for it in ref[task]["items"]}
    qp = {it["id"]: it["pred"] for it in ref[task]["items"]}
    for a, b in (("ocean", "r4"), ("r8", "r4"), ("ocean", "r8"), ("ocean", "qg4 bench"), ("r4", "qg4 bench")):
        n, pa, pb, wa, wb, z = paired(ok[a], q if b == "qg4 bench" else ok[b])
        rows.append((task, a, b, n, pa, pb, wa, wb, z))
    agree = [it["pred"] == qp[it["id"]] for it in g["r4"][task]["items"] if it["id"] in qp]
    print(f"{task}: r4 vs qg4 bench, same answer on {sum(agree)}/{len(agree)} ({sum(agree) / len(agree):.1%})")

print("\n| task | A | B | n | acc A | acc B | A-B (points) | A right, B wrong | B right, A wrong | z |")
print("|---|---|---|---|---|---|---|---|---|---|")
for task, a, b, n, pa, pb, wa, wb, z in rows:
    print(f"| {task} | {a} | {b} | {n} | {pa * 100:.1f} | {pb * 100:.1f} | {(pa - pb) * 100:+.1f} | {wa} | {wb} | {z:+.2f} |")
