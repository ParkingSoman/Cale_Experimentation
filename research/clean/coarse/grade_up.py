"""Grade the UPSTREAM PARENT test (judge_out3.json vs key_up.json). Pass: real >= 50% and >= 2x decoys."""
import json, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "poles")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
k = json.load(open(os.path.join(D, "key_up.json"))); j = json.load(open(os.path.join(D, "judge_out3.json")))
d = json.load(open(os.path.join(D, "desc_up.json")))
r, dc, per = [], [], {}
for p in k["pairs"]:
    yes = str(j[p["id"]]["answer"]).lower().startswith("yes")
    (r if p["real"] else dc).append(yes)
    if p["real"]: per[p["child"]] = {"grouping": p["grouping"], "parent": k["info"][p["child"]]["parent"], "yes": yes,
                                     "parent_desc": d[p["unit"]]["function"][:160]}
res = {"real_yes": sum(r) / len(r), "decoy_yes": sum(dc) / len(dc), "n_real": len(r), "per_child": per}
res["pass"] = bool(res["real_yes"] >= 0.5 and res["real_yes"] >= 2 * res["decoy_yes"])
json.dump(res, open(os.path.join(ROOT, "results", "clean-upstream-judge.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
