"""Combine the input axis (r1cover) and the output axis (r1out), one score per unit.

Per unit: best pole (the pole with more children) on each axis, then the better axis.
The same rule is applied to the two baselines (untrained random neurons, old neurons as parents).
  python3 research/clean/r1combine.py
Writes results/clean-r1combine.json.
"""
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(ROOT, "results")


def per_unit(units):
    best = defaultdict(lambda: {"n_children": 0, "up_cover": 0.0})
    for u in units:
        b = best[u["unit"]]
        if (u["n_children"], u["up_cover"]) > (b["n_children"], b["up_cover"]):
            best[u["unit"]] = {"n_children": u["n_children"], "up_cover": u["up_cover"]}
    return dict(best)


def score(axes):
    """axes: list of per-unit dicts (same unit ids). Returns share stats over units, best axis per unit."""
    ids = set().union(*[a.keys() for a in axes])
    best = [max((a.get(i, {"n_children": 0, "up_cover": 0.0}) for a in axes),
                key=lambda b: (b["n_children"], b["up_cover"])) for i in ids]
    n = len(best)
    return {"units": n, "share_2plus": sum(b["n_children"] >= 2 for b in best) / n,
            "share_upcover_50": sum(b["up_cover"] >= 0.5 for b in best) / n,
            "share_both": sum(b["n_children"] >= 2 and b["up_cover"] >= 0.5 for b in best) / n}


def main():
    cov = {(r["layer"], r["K"], r["tag"]): r for r in json.load(open(os.path.join(R, "clean-r1cover.json")))["rows"]}
    out = {(r["layer"], r["K"], r["tag"]): r for r in json.load(open(os.path.join(R, "clean-r1out.json")))["rows"]}
    dp = os.path.join(R, "clean-r1out-downstream.json")
    down = {(r["layer"], r["K"], r["tag"]): r for r in json.load(open(dp))["rows"]} if os.path.exists(dp) else {}
    res = []
    print(f"{'layer K tag':22s} {'kind':12s} {'axis':10s} {'2+ children':>12s} {'up-cover>=.5':>13s} {'both':>6s}")
    for key in sorted(set(cov) | set(out) | set(down)):
        for kind, field in (("trained", "units"), ("random", "null_random_units"), ("old", "null_old_units")):
            axes = {}
            for name, src in (("input", cov), ("output", out), ("downstream", down)):
                if key in src and field in src[key]:
                    axes[name] = per_unit(src[key][field])
            if not axes:
                continue
            for name, sel in list((k, [v]) for k, v in axes.items()) + ([("any", list(axes.values()))]
                                                                         if len(axes) >= 2 else []):
                s = score(sel)
                res.append({"layer": key[0], "K": key[1], "tag": key[2], "kind": kind, "axis": name, **s})
                print(f"L{key[0]:<2d} K={key[1]:<3d} {key[2]:11s} {kind:12s} {name:10s} {s['share_2plus']:12.2f} "
                      f"{s['share_upcover_50']:13.2f} {s['share_both']:6.2f}")
    json.dump(res, open(os.path.join(R, "clean-r1combine.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
