"""D2 look at the threshold: firing rates of trained narrow neurons vs their parts; built net's score as the rate
multiplier changes (description, to see the shape, not to pick a value)."""
import os, sys
import numpy as np
from d2_lib import *
W = net("gelu_256_s0"); Nn = net("gelu_16_s0"); Z = Xr @ W["W1"].T + W["b1"]; Hr = gl(Z); ZN = Xr @ Nn["W1"].T + Nn["b1"]
zs = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-9); Cc = zs(Hr).T @ zs(gl(ZN)) / len(Hr); parts = [np.argsort(-Cc[:, j])[:10] for j in range(16)]
print("trained narrow firing rate (z>0) median %.3f; its parts' mean rate median %.3f; wide neurons' rate median %.3f" % (
    np.median((ZN > 0).mean(0)), np.median([(Z[:, p] > 0).mean() for p in parts]), np.median((Z > 0).mean(0))))
gs = groups_by(Hr, 16, "ward")
for m in (0.25, 0.5, 1.0, 2.0, 3.0, 4.0):
    n = build_w(W, gs, [np.ones(len(g)) for g in gs], m); ZB = Xr @ n["W1"].T + n["b1"]
    print(f"  rate x{m}: built rate median {np.median((ZB > 0).mean(0)):.3f}  fine %.3f coarse %.3f KL %.3f" % score(bayes_write(W, n), W))
