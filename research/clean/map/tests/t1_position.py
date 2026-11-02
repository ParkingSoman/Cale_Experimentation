"""T1: where in the frame does each neuron fire? Mean map, mean positive part, argmax-position histogram (3000 imgs)."""
from mid_common import *
G._init(); X = np.asarray(G._X[G._idx]) if False else None
Z = zmaps(G._X[G._idx])
np.set_printoptions(precision=2, suppress=True, linewidth=200)
for t, z in Z.items():
    mxv = z.reshape(len(z), -1).max(1); am = z.reshape(len(z), -1).argmax(1)
    top = np.argsort(-mxv)[:300]; h = np.bincount(am[top], minlength=196).reshape(14, 14)
    print(t, "max", mxv.max().round(2), "on-rate(z>0)", (z > 0).mean().round(3))
    print(" mean relu map rows (top->bottom):", np.maximum(z, 0).mean((0, 2)))
    print(" mean relu map cols (left->right):", np.maximum(z, 0).mean((0, 1)))
    print(" argmax rows of top-300 imgs:", h.sum(1)); print(" argmax cols:", h.sum(0))
