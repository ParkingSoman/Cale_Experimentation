"""CRITIC round 3: paired read of two full board rows on the 800 test photos NOT in the quick set (no torch).

Object (picked by): two saved per-photo files of full board runs, fanout/out/<tag>.npy and <tag>_agree.npy
(picked by the claim under test). Photo use: test photos, scoring only (reads saved KL and agree).
Quick set = index % 5 == 0. Prints mean KL and agree of A and B on quick200 and other800, the paired KL
difference B - A with SE on the 800, and discordant agree counts with McNemar z on the 800.
Usage: curves/.venv/bin/python -I distill/fanout/r3_critic_holdout.py A_tag B_tag [A2 B2 ...]   (tag = file name without .npy)
"""
import os
import sys

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def load(tag):
    tag = tag[:-4] if tag.endswith(".npy") else tag
    k = np.load(os.path.join(OUT, tag + ".npy"))
    p = os.path.join(OUT, tag + "_agree.npy")
    g = np.load(p).astype(bool) if os.path.exists(p) else None
    return k, g


args = sys.argv[1:]
for a, b in zip(args[::2], args[1::2]):
    kA, gA = load(a)
    kB, gB = load(b)
    q = np.arange(len(kA)) % 5 == 0
    d = (kB - kA)[~q]
    s = (f"A={a}\n B={b}\n  KL quick200 A {kA[q].mean():.4f} B {kB[q].mean():.4f} | other800 A {kA[~q].mean():.4f} "
         f"B {kB[~q].mean():.4f} | B-A on 800 {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}")
    if gA is not None and gB is not None:
        n10 = int((gA & ~gB)[~q].sum()); n01 = int((~gA & gB)[~q].sum())
        z = (n01 - n10) / max(1, np.sqrt(n01 + n10))
        s += (f"\n  agree quick200 A {gA[q].mean():.3f} B {gB[q].mean():.3f} | other800 A {gA[~q].mean():.3f} B {gB[~q].mean():.3f}"
              f" | 800: A-only {n10}, B-only {n01}, McNemar z {z:+.2f}")
    else:
        s += "\n  (no per-photo agree file for one of the two)"
    print(s)
