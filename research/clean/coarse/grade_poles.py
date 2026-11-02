"""Grade the POLES test from judge_out.json against key.json. Rules: CHILDREN-LOG 'POLES test'."""
import json, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "poles")
k = json.load(open(os.path.join(D, "key.json"))); j = json.load(open(os.path.join(D, "judge_out.json")))
A = {p["id"]: p for p in k["pairs"]}; B = {t["id"]: t for t in k["triples"]}
ra = {c: None for c in k["stats"]}; real_a, dec_a = [], []
for i, p in A.items():
    yes = str(j["A"][i]["answer"]).lower().startswith("related")
    (real_a if p["real"] else dec_a).append(yes)
    if p["real"]: ra[p["child"]] = yes
real_b, dec_b, rb = [], [], {}
for i, t in B.items():
    yes = str(j["B"][i]["answer"]).lower().startswith("yes")
    (real_b if t["real"] else dec_b).append(yes)
    if t["real"]: rb[t["child"]] = yes
a_kids = [c for c, v in ra.items() if v]
b_on_a = [rb[c] for c in a_kids]
res = {"A_real_related": sum(real_a) / len(real_a), "A_decoy_related": sum(dec_a) / len(dec_a),
       "B_real_yes": sum(real_b) / len(real_b), "B_decoy_yes": sum(dec_b) / len(dec_b),
       "B_real_yes_among_A_related": (sum(b_on_a) / len(b_on_a)) if b_on_a else None,
       "children_A_related": a_kids, "children_B_yes": [c for c, v in rb.items() if v],
       "gate_open_share": {c: round(v["gate_open_share"], 3) for c, v in k["stats"].items()}}
res["rule_a"] = bool(res["A_real_related"] >= 0.5 and res["A_real_related"] >= 2 * res["A_decoy_related"])
res["rule_b"] = bool(b_on_a and res["B_real_yes_among_A_related"] >= 0.5 and res["B_real_yes_among_A_related"] >= 2 * res["B_decoy_yes"])
json.dump(res, open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(D))), "..", "results", "clean-poles-test.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
