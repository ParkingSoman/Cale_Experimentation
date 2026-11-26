"""Hubness (description): how often each entry is the top-1 reading for 2000 random write-like vectors and 2000 random
read-like vectors, in the tool metric. Uniform = 1/N per entry."""
import sys
from collections import Counter

import numpy as np

import vecread as vr

rng = np.random.default_rng(3)
for space in sys.argv[1:] or vr.SPACES:
    dic = vr.Dictionary(space)
    D = dic.unit(True)
    names = dic.names[dic.kept]
    N = len(D)
    for kind in ("write", "read"):
        G = rng.standard_normal((2000, dic.C))
        if kind == "read":
            G = G @ dic.cov
        U = dic.metric(G, True)
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        top = np.argmax(U @ D.T, axis=1)
        c = Counter(top)
        share = np.array(sorted(c.values(), reverse=True)) / 2000
        print(f"{space} {kind}: N={N}, entries ever top-1 {len(c)}, top-5 shares {np.round(share[:5], 3).tolist()} "
              f"(uniform {1 / N:.4f}): {[str(names[k]) for k, _ in c.most_common(5)]}")
