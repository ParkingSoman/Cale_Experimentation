"""Score judge answers against the key. Usage: python3 score_judge.py s2 A '<json>' B '<json>'
Hit rate = share of the judge's 24 picks that are real targets (chance .50). Real words vs control words per sheet."""
import json, os, sys
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", sys.argv[1])
key = json.load(open(os.path.join(OUT, "_key", "key.json")))
rows = []
for j, ans in zip(sys.argv[2::2], sys.argv[3::2]):
    ans = json.loads(ans)
    for t in json.load(open(os.path.join(OUT, "_key", f"judge_{j}_tasks.json"))):
        picks = set(ans[str(t["task"])])
        tgt = set(key["J_" + t["sheet_id"]])
        rows.append((t["sheet_id"], t["kind"], j, len(picks), len(picks & tgt) / max(len(picks), 1)))
res = {}
for s in sorted({r[0] for r in rows}):
    line = {}
    for kind in ("real", "control"):
        v = [r[4] for r in rows if r[0] == s and r[1] == kind]
        line[kind] = [round(x, 2) for x in v]
    res[s] = line
    print(f"{s:6s} real words: {line['real']}   control words: {line['control']}   (chance .50)")
bad = [r for r in rows if r[3] != 24]
if bad: print("WARNING: answers without 24 picks:", bad)
json.dump(res, open(os.path.join(OUT, "judge_scores.json"), "w"), indent=1)
