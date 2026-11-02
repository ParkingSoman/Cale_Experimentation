"""Heat sheets: neuron's map on its top analysis images of chosen classes. usage: python last_heat.py B N fname class1,class2,..."""
import sys, numpy as np
from last_lib import *
b, n, fname, cls = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4].split(",")
X, y, ana = data(); nm = names(); v = amax(b, n); ya = y[ana]; idx = []
for c in cls:
    ci = nm.index(c); j = np.where(ya == ci)[0]; idx += [int(ana[k]) for k in j[np.argsort(-v[j])[:2]]]
print(heat_sheet(b, n, idx, fname, f"block {b} n{n}"))
