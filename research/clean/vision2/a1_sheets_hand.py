import json, os
import numpy as np
from gallery_x import *
A1 = json.load(open(os.path.join(RES, "vision-a1-small.json"))); ids = {k: [tuple(x) for x in v] for k, v in A1["ids"].items()}
HAND = [(14, 1326, 13, 78), (16, 384, 15, 1162), (16, 26, 15, 2142), (17, 2283, 16, 2886), (11, 1031, 10, 1172), (10, 1316, 9, 686)]
small = "atto"; nb = {"atto": (2, 2, 6, 2)}
off = {2: 2 + 2, 3: 2 + 2 + 6}   # first atto block of stage 3 / 4
for cb, cn, pb, pn in HAND:
    st = "stage4" if cb >= 15 else "stage3"; j = ids[st].index((pb, pn)); jc = ids[st].index((cb, cn))
    arg = A1[small]["best_match"][st]["arg"][j]; r = A1[small]["best_match"][st]["r"][j]; rc = A1[small]["best_match"][st]["r"][jc]; argc = A1[small]["best_match"][st]["arg"][jc]
    m = model(small); W = [m.blocks[b].fc1.weight.shape[0] for b in range(len(m.blocks))]; s0 = off[3 if cb >= 15 else 2]
    b = s0; a = arg
    while a >= W[b]: a -= W[b]; b += 1
    print((cb, cn, pb, pn), "parent's atto match", (b, a), "r", round(r, 2), "| child's own best r", round(rc, 2), "same neuron" if argc == arg else "different")
    print(family_sheet((pb, pn), (cb, cn), small, (b, a), f"a1/hand_{cb}_{cn}.png", f"tiny parent {pb} n{pn} -> child {cb} n{cn}; atto match of the parent: block {b} n{a} (r {r:.2f}); child's own best r {rc:.2f}"))
