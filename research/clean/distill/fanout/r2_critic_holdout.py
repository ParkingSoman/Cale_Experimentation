"""CRITIC round 2: read a promoted gain on the 800 full-run photos that are NOT in the quick set (no torch).

Object (picked by): two saved per-photo KL files of full board runs, fanout/out/*.npy (picked by the claim).
Photo use: test photos, scoring only (reads saved KL). The quick set is index % 5 == 0; the rest were never seen by
any quick-loop choice. Prints mean KL of A and B on the 200 quick photos and on the 800 others, and the paired
difference B - A with its SE on the 800.
Usage: curves/.venv/bin/python -I distill/fanout/r2_critic_holdout.py A.npy B.npy [A2.npy B2.npy ...]
"""
import os
import sys

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
args = sys.argv[1:]
for a, b in zip(args[::2], args[1::2]):
    kA, kB = np.load(os.path.join(OUT, a)), np.load(os.path.join(OUT, b))
    q = np.arange(len(kA)) % 5 == 0
    d = (kB - kA)[~q]
    print(f"A={a} B={b}: quick200 A {kA[q].mean():.4f} B {kB[q].mean():.4f} | other800 A {kA[~q].mean():.4f} "
          f"B {kB[~q].mean():.4f} | B-A on 800 {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}")
