# scale: neurons=every node of CLIP ViT-B/16 and ResNet-18; net=both; photos=none (generic input only); where=mac
# rung: Q=tree step 0 (branch statistics); EST=2 min; PERMIT=Shonu approved plan 2026-11-26 late night
"""Step 0 (plan 2026-11-26 late night): the tree's shape per block, from weights + the generic input. No photos.
Per block: distinct main parents, share of nodes with a sibling, ancestor-chain depth, main-parent clarity (top / second
score), group count at levels 1-6 and at the roots, and the level the cut reaches at 95 / 85 / 50 / 25 / 5%.
Usage: curves/.venv/bin/python -I distill/tree/step0.py clip|r18
Output: distill/tree/out/branches_<net>.json
"""
import json
import os
import statistics
import sys
import time
from collections import Counter

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DIST = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DIST, "fanout"))
import build as TB  # noqa: E402


def main():
    net = sys.argv[1]
    t0 = time.time()
    X = torch.load(os.path.join(DIST, "fanout", "cache", f"{net}_self_n256_k32_s1.pt"), weights_only=True).float()
    if net == "clip":
        import board
        T = board.clip_teacher()
        st = TB.clip_stats(T, X)
        tr = TB.clip_tree(T, st)
        kinds = ("h", "m")
    else:
        import board
        A = board.r18_A()
        st = TB.r18_stats(A, X)
        tr = TB.r18_tree(A, st)
        kinds = ("m",)
    names, parent = tr["names"], tr["parent"]
    nodes = [g for g in range(len(names)) if names[g][0] != "e"]
    ch = TB.chains(parent, nodes)
    rows = []
    for b in range(tr["nb"]):
        for kind in kinds:
            n = tr["strength"][(kind, b)].numel()
            ids = [tr["gid"][(kind, b, i)] for i in range(n)]
            par = Counter(parent[g] for g in ids)
            pk = Counter(names[p][0] + ("" if names[p][0] == "e" else str(names[p][1])) for p in par.elements())
            row = dict(block=b, kind=kind, n=n, parents=len(par),
                       sibling_share=round(sum(par[parent[g]] > 1 for g in ids) / n, 3),
                       biggest_family=max(par.values()),
                       parent_kinds=dict(pk.most_common(6)),
                       depth_median=statistics.median(len(ch[g]) for g in ids),
                       depth_max=max(len(ch[g]) for g in ids),
                       clarity_median=round(statistics.median(tr["clarity"][g] for g in ids), 3),
                       clarity_share_below_1_1=round(sum(tr["clarity"][g] < 1.1 for g in ids) / n, 3))
            sub = {g: ch[g] for g in ids}
            row["groups_at_level"] = {str(L): len(TB._partition(ids, sub, L)) for L in (1, 2, 3, 4, 6, 10)}
            row["groups_at_roots"] = len(TB._partition(ids, sub, 999))
            lv = {}
            for f in (0.95, 0.85, 0.5, 0.25, 0.05):
                k = max(1, round(f * n))
                _, info = TB.cut(ids, sub, k, tr["strength"][(kind, b)], tr["reads"][(kind, b)])
                lv[str(f)] = dict(level=info["level"], max_group=info.get("max_group", 1), singles=info.get("singles", n))
            row["cut"] = lv
            rows.append(row)
            print(json.dumps(row), flush=True)
    out = os.path.join(HERE, "out", f"branches_{net}.json")
    json.dump(dict(net=net, input=f"{net}_self_n256_k32_s1", photo_use="none", rows=rows, seconds=round(time.time() - t0, 1)),
              open(out, "w"), indent=1)
    print("saved", out, round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
