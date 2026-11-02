"""Grade the REVISED POLES test (judge_out2.json vs key_region.json). Pass: Part B real yes >= 50% and >= 2x decoys."""
import json, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "poles")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
k = json.load(open(os.path.join(D, "key_region.json"))); j = json.load(open(os.path.join(D, "judge_out2.json")))
def rate(items, part):
    r, d, per = [], [], {}
    for it in items:
        yes = str(j[part][it["id"]]["answer"]).lower().startswith("yes")
        (r if it["real"] else d).append(yes)
        if it["real"]: per[it["child"]] = (yes, j[part][it["id"]].get("category", ""))
    return sum(r) / len(r), sum(d) / len(d), per
a_r, a_d, a_per = rate(k["pairs"], "A"); b_r, b_d, b_per = rate(k["triples"], "B")
res = {"A_real_yes": a_r, "A_decoy_yes": a_d, "B_real_yes": b_r, "B_decoy_yes": b_d,
       "A_groupings_real": {c: v[1] for c, v in a_per.items() if v[0]}, "B_yes_children": [c for c, v in b_per.items() if v[0]]}
res["pass"] = bool(b_r >= 0.5 and b_r >= 2 * b_d)
json.dump(res, open(os.path.join(ROOT, "results", "clean-poles-revised.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
