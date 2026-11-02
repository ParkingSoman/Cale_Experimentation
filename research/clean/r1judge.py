"""Blind scoring and judging for r1family.py outputs.

  python3 research/clean/r1judge.py prep  L3_input     # writes *_score_in.md and *_judge_in.md (+ hidden *_judge_key.json)
  python3 research/clean/r1judge.py grade L3_input     # reads *_score_out.json and *_judge_out.json
Grades go to results/clean-r1judge-<stem>.json.

Scoring (delphi detection): the scorer sees one parent description and 10 items, picks the 5 where it is active.
Judging: the judge sees one parent description and two sets of child descriptions (real children vs decoy old
neurons, order random) and picks the set the parent is a broader category of.
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "r1family")
ROOT = os.path.dirname(os.path.dirname(HERE))


def load(stem):
    key = json.load(open(os.path.join(D, f"{stem}_key.json")))
    desc = json.load(open(os.path.join(D, f"{stem}_desc.json")))
    return key, desc


def prep(stem):
    key, desc = load(stem)
    rng = np.random.default_rng(7)
    det = open(os.path.join(D, f"{stem}_detect.md")).read()
    blocks = dict(re.findall(r"\n## (U\d+)\n(.*?)(?=\n## |\Z)", det, flags=re.S))
    s = [f"# Detection scoring ({stem})\n", "For each unit: read the description. Then pick EXACTLY 5 of the 10 items",
         "where you think the unit is active. Items are text contexts (the active token is in [[ ]]) or words.\n"]
    for uid in key["detect_key"]:
        s += [f"\n## {uid}\n", f"Description: {desc[uid]['function']}", "Items:", blocks[uid].strip()]
    open(os.path.join(D, f"{stem}_score_in.md"), "w").write("\n".join(s) + "\n")
    j = [f"# Family judging ({stem})\n",
         "Each case has one PARENT unit and two sets of other units (X and Y). Exactly one set holds units that the",
         "parent was found to cover. Pick the set for which the parent is best described as a broader, coarser",
         "category that includes the units in the set. Give a confidence 1 (guess) to 5 (sure).\n"]
    jk = {}
    for i, f in enumerate(key["families"]):
        real = [desc[u]["function"] for u in f["children"]]
        fake = [desc[u]["function"] for u in f["decoys"]]
        flip = bool(rng.integers(2))
        X, Y = (fake, real) if flip else (real, fake)
        jk[f"F{i + 1}"] = "Y" if flip else "X"
        j += [f"\n## F{i + 1}\n", f"PARENT: {desc[f['parent']]['function']}", "Set X:"] + [f"  - {x}" for x in X]
        j += ["Set Y:"] + [f"  - {y}" for y in Y]
    open(os.path.join(D, f"{stem}_judge_in.md"), "w").write("\n".join(j) + "\n")
    json.dump(jk, open(os.path.join(D, f"{stem}_judge_key.json"), "w"))
    print(f"prepared {stem}: {len(key['detect_key'])} scoring units, {len(jk)} families")


def grade(stem):
    key, desc = load(stem)
    so = json.load(open(os.path.join(D, f"{stem}_score_out.json")))
    jo = json.load(open(os.path.join(D, f"{stem}_judge_out.json")))
    jk = json.load(open(os.path.join(D, f"{stem}_judge_key.json")))
    acc = []
    for uid, truth in key["detect_key"].items():
        picks = set(int(x) for x in so.get(uid, []))
        acc.append(np.mean([(i + 1 in picks) == t for i, t in enumerate(truth)]))
    # Families with no children are not graded (the judge can only guess there).
    has = {f"F{i + 1}" for i, f in enumerate(key["families"]) if f["children"]}
    right = [jo[f]["choice"] == jk[f] for f in jk if f in jo and f in has]
    conf_right = [jo[f]["confidence"] for f in jk if f in jo and f in has and jo[f]["choice"] == jk[f]]
    kinds = {k: [desc[u]["coherent"] for u, v in key["units"].items() if v["kind"] == k and u in desc]
             for k in ("parent", "child", "decoy")}
    res = {"stem": stem, "detection_accuracy_mean": float(np.mean(acc)), "detection_per_parent": [float(a) for a in acc],
           "judge_right": int(sum(right)), "judge_n": len(right), "judge_right_confidence_mean":
           float(np.mean(conf_right)) if conf_right else None,
           "share_coherent": {k: float(np.mean(v)) if v else None for k, v in kinds.items()}}
    # One-sided binomial p-value for the judge against chance (0.5).
    from math import comb
    n, k = res["judge_n"], res["judge_right"]
    res["judge_p_value"] = float(sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)
    json.dump(res, open(os.path.join(ROOT, "results", f"clean-r1judge-{stem}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "detection_per_parent"}, indent=1))


if __name__ == "__main__":
    {"prep": prep, "grade": grade}[sys.argv[1]](sys.argv[2])
