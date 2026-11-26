"""Pool two sheets of one round pair (e.g. w1 + w2) per pole from results/judge_score.json (run oc_score.py first).
Prints own and control precision over 24 targets, the gap, a rough standard error, and the bars (B1 >= .75,
B2 gap >= .3). Usage: python3 oc_pool.py w1 w2"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "..", "results", "judge_score.json")))
a, b = sys.argv[1], sys.argv[2]
print(f"{'pole':14s} {'own':>5s} {'ctrl':>5s} {'gap':>6s} {'se':>5s}  B1    B2")
for n in ["p", "k275", "k297", "k97", "k305", "k163", "k23", "c101"]:
    for side, ka, kb in (("positive", f"{n}_{a}", f"{n}_{b}"), ("negative", f"{n}_neg{a}", f"{n}_neg{b}")):
        if ka not in R or kb not in R:
            continue
        own = (R[ka]["own"] + R[kb]["own"]) / 2
        ctrl = (R[ka]["ctrl"] + R[kb]["ctrl"]) / 2
        se = np.sqrt(max(own * (1 - own), .05) / 24)
        print(f"{n + ' ' + side:14s} {own:5.2f} {ctrl:5.2f} {own - ctrl:+6.2f} {se:5.2f}  {'pass' if own >= .75 else 'FAIL'}  {'pass' if own - ctrl >= .3 else 'FAIL'}")
