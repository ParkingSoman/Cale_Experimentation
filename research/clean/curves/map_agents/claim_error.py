# serves: H-map
"""The code that judges a map agent's claim. It trusts nothing the agent says. It computes errors from N, P and the claim.

Input: a JSON file with a list of claims. Each claim:
  {"name": "...", "units": [channels of the last hidden layer],
   "region": {"gate": {"units": [...], "weights": [...], "theta": number}}     a photo is inside when the read of the channels
                                                                               is above theta (so the region is the network's own)
             or {"photos": [indices into P]} or {"all": true}}

Output per claim, with the definitions of COARSE-THEORY.md A.7:
  eps_dist      DM2: largest member's share of its own activity outside the region
  eps_dist_mean the same, mean over members
  eps_quiet     DM9, as a share: what the members write to the logits outside the region (mean osc) / what they write on all of P
  kl_cut        KL of N when the members are set to 0 outside the region (N's readers kept): the real cost of the claim
  kl_drop       KL of N when the members are set to 0 everywhere
  mass          the region's share of P
  own_inside    on the region, the members' share of the Fisher-weighted logit movement (how much of N's decision they carry)
Usage: claim_error.py NET claims.json [--out result.json]"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "cache", "map")
ap = argparse.ArgumentParser()
ap.add_argument("net")
ap.add_argument("claims")
ap.add_argument("--out")
a = ap.parse_args()
meta = np.load(os.path.join(OUT, f"{a.net}_meta.npz"))
W = meta["W"].astype(np.float64)
H = np.load(os.path.join(OUT, f"{a.net}_h.npy")).astype(np.float64)
Y = np.load(os.path.join(OUT, f"{a.net}_y.npy")).astype(np.float64)
n = H.shape[0]
Wc = W - W.mean(0, keepdims=True)


def softmax(y):
    y = y - y.max(1, keepdims=True)
    e = np.exp(y)
    return e / e.sum(1, keepdims=True)


S = softmax(Y)
lS = np.log(S + 1e-300)


def kl(y2):
    return float((S * (lS - np.log(softmax(y2) + 1e-300))).sum(1).mean())


def region(r):
    if r.get("all"):
        return np.ones(n, bool)
    if "photos" in r:
        m = np.zeros(n, bool)
        m[np.asarray(r["photos"], int)] = True
        return m
    g = r["gate"]
    return H[:, g["units"]] @ np.asarray(g["weights"], float) > g["theta"]


rows = []
for c in json.load(open(a.claims)):
    U = c["units"]
    R = region(c["region"])
    h = H[:, U]
    A = h.sum(0)
    eps_u = np.where(A > 0, h[~R].sum(0) / np.where(A > 0, A, 1), 0)
    write = h @ W[:, U].T
    osc = write.max(1) - write.min(1)
    eps_q = float(osc[~R].sum() / osc.sum()) if osc.sum() > 0 else 0.0
    cut = Y.copy()
    cut[~R] -= write[~R]
    # the members' share of the Fisher-weighted movement on the region (about each photo's region mean)
    if R.sum() > 1:
        Yr = Y[R] - Y[R].mean(0)
        wr = write[R] - write[R].mean(0)
        Sr = S[R]
        var = lambda d: (Sr * d ** 2).sum(1) - ((Sr * d).sum(1)) ** 2
        own = float(var(wr).sum() / max(var(Yr).sum(), 1e-12))
    else:
        own = 0.0
    rows.append(dict(name=c.get("name", ""), n_units=len(U), mass=float(R.mean()), eps_dist=float(eps_u.max()),
                     eps_dist_mean=float(eps_u.mean()), eps_quiet=eps_q, kl_cut=kl(cut), kl_drop=kl(Y - write),
                     own_inside=own))
    print(json.dumps(rows[-1]))
if a.out:
    json.dump(rows, open(a.out, "w"), indent=1)
