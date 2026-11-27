"""Result 2 (ResNet-18 ladder, research/clean/onecoarse/VECMATH.md section 7): print the tables from the saved results.

Needs numpy only. No photos, no GPU, a few seconds.
Step 1 runs the original reader, research/clean/onecoarse/code/oc_ladder_read.py, on the committed folder
research/clean/onecoarse/results/oc_ladder. It prints the recovery table (as saved in each oc_ladder.json) and the
paired-bootstrap differences (computed from the per-photo arrays kl_test_*.npy and strong.npy).
Step 2 computes recovery again from the per-photo arrays, recovery = 1 - mean KL(build) / mean KL(Z) on all test
photos, and checks it against the saved table.

Usage (from the repository root): python repro/ladder_tables.py
"""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "research", "clean", "onecoarse", "code")
RES = os.path.join(ROOT, "research", "clean", "onecoarse", "results", "oc_ladder")

print("== Step 1: oc_ladder_read.py ==", flush=True)
subprocess.run([sys.executable, os.path.join(CODE, "oc_ladder_read.py"), RES], check=True)

print("\n== Step 2: recovery on all test photos, from the per-photo arrays ==")
print("| Rung | C (calculation) | R4 (training) | R4C (calculation, then training) | matches saved |")
print("|---|---|---|---|---|")
ok_all = True
for n in (11, 64, 192, 384):
    d = os.path.join(RES, f"n{n}")
    saved = json.load(open(os.path.join(d, "oc_ladder.json")))["builds"]
    z = np.load(os.path.join(d, "kl_test_Z.npy")).astype(np.float64)
    got = {b: 1 - np.load(os.path.join(d, f"kl_test_{b}.npy")).astype(np.float64).mean() / z.mean()
           for b in ("C", "R4", "R4C")}
    ok = all(abs(got[b] - saved[b]["rec_test"]) < 5e-4 for b in got)
    ok_all &= ok
    print(f"| {n} | {got['C']:.3f} | {got['R4']:.3f} | {got['R4C']:.3f} | {'yes' if ok else 'NO'} |")
print("\nAll rungs match the saved table." if ok_all else "\nSOME RUNGS DO NOT MATCH the saved table.")
sys.exit(0 if ok_all else 1)
