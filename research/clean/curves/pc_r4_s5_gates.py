# serves: TQ11
"""Round 4, screen S5 (card 3-37): does the gate help the trunk's real box?

Object: V_b(k) on ResNet-18's last block. A unit is relu(m . z(x, s) + beta) at each of 49 positions, then the mean over
positions (Definition DY1). Logits t + V c(x). Compared with the always-on read L_on(k) on the same rows.
"Gates do not help" was withdrawn in LADDER 49: no earlier run tested it (one stalled at 60 iterations).

Runs, all float64, same rows, same stopping rule:
  read:   L_on, from the PCA span.
  warm:   box from the read, with each beta set so the unit is on at about half of the positions.
  rand:   box from a random start, two seeds. A floor needs real starts (feedback-math-method 5b).

Screen line, fixed before the run: V_b is an upper value, so a gain above 10% against the read (train and sealed) means the
gate matters, and S2's target is the box and not the read. A gain below 10% keeps the read as the object.

PREDICTION, written before the first run (2026-11-02): the gain on train is between 3% and 15%, and the gain on sealed
photos is smaller than on train. I expect the warm start to beat the random starts by under 5%.

Usage: pc_r4_s5_gates.py [--smoke] [--n 8000] [--k 14] [--chunks 40]
Stopping rule: chunks of 50 LBFGS iterations; stop when one chunk gains under 2e-5 nats (or the chunk limit, reported).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--n", type=int, default=8000)
ap.add_argument("--k", type=int, default=14)
ap.add_argument("--chunks", type=int, default=40)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "rounds/round4")
torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)
n = 400 if a.smoke else a.n
ns = 200 if a.smoke else 3000
chunks = 3 if a.smoke else a.chunks


def load(name, count):
    Z = np.load(os.path.join(HERE, f"cache/r18/z_{name}.npy"), mmap_mode="r")
    Y = np.load(os.path.join(HERE, f"cache/r18/y_{name}.npy"), mmap_mode="r")
    return (torch.from_numpy(np.asarray(Z[:count]).astype(np.float64)).permute(0, 2, 1).contiguous(),
            torch.from_numpy(np.asarray(Y[:count]).astype(np.float64)))


Z, Y = load("train", n)
Z2, Y2 = load("sealed", ns)
mu, sd = Z.mean((0, 1)), Z.std((0, 1)) + 1e-4
Z, Z2 = (Z - mu) / sd, (Z2 - mu) / sd
Zp, Zp2 = Z.mean(1), Z2.mean(1)
ls, ls2 = torch.log_softmax(Y, 1), torch.log_softmax(Y2, 1)
S, S2 = ls.exp(), ls2.exp()


def kl(L, S, ls):
    return (S * (ls - torch.log_softmax(L, 1))).sum(1).mean()


def code(Z, M, b):
    return torch.relu(Z @ M + b).mean(1)


def fit(params, fn, label):
    """LBFGS in chunks, with the value logged, so a stall is visible."""
    last = float(fn().detach())
    hist = [last]
    t0 = time.time()
    stop = "chunk limit"
    for c in range(chunks):
        opt = torch.optim.LBFGS(params, lr=1, max_iter=50, history_size=30, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-9, tolerance_change=1e-12)

        def cl():
            opt.zero_grad()
            v = fn()
            v.backward()
            return v
        opt.step(cl)
        now = float(fn().detach())
        hist.append(now)
        print(f"{label} chunk {c} value {now:.6f} sec {time.time() - t0:.0f}", flush=True)
        if last - now < 2e-5:
            stop = "gain under 2e-5"
            break
        last = now
    return hist, stop


k = a.k
Yc = Y - Y.mean(1, keepdim=True)
ybar = Yc.mean(0)
D = Yc - ybar
U = torch.linalg.eigh(D.T @ D / n)[1].flip(1)
Q0 = U[:, :k].contiguous()
results = {"n": n, "ns": ns, "k": k, "runs": {}}

# read
M = torch.linalg.lstsq(Zp, D @ Q0).solution.contiguous().requires_grad_(True)
V = Q0.clone().requires_grad_(True)
t = ybar.clone().requires_grad_(True)
hist, stop = fit([M, V, t], lambda: kl(t + (Zp @ M) @ V.T, S, ls), "read")
read_train = hist[-1]
with torch.no_grad():
    read_sealed = float(kl(t + (Zp2 @ M) @ V.T, S2, ls2))
results["runs"]["read"] = dict(train=read_train, sealed=read_sealed, stop=stop, hist=hist)
print("read", read_train, read_sealed, flush=True)
M_read, V_read, t_read = M.detach().clone(), V.detach().clone(), t.detach().clone()


def box_run(label, M0, V0, b0, t0):
    M, V, b, t = [x.clone().requires_grad_(True) for x in (M0, V0, b0, t0)]
    hist, stop = fit([M, V, b, t], lambda: kl(t + code(Z, M, b) @ V.T, S, ls), label)
    with torch.no_grad():
        sealed = float(kl(t + code(Z2, M, b) @ V.T, S2, ls2))
        on = float(((Z @ M + b) > 0).double().mean())
    results["runs"][label] = dict(train=hist[-1], sealed=sealed, on=on, stop=stop, hist=hist)
    print(label, hist[-1], sealed, "on", on, flush=True)


# warm: rescale nothing, set beta so each unit is on at about half of the positions, and move the mean into t
with torch.no_grad():
    b0 = -(Z @ M_read).reshape(-1, k).median(0).values
    # the relu of a half-on unit has about half the slope, so double the read to keep the code's scale
    t0 = t_read - (2 * code(Z, M_read, b0).mean(0) - (Zp @ M_read).mean(0)) @ V_read.T
box_run("warm", 2 * M_read, V_read, 2 * b0, t0)

for seed in (1, 2):
    torch.manual_seed(seed)
    M0 = torch.randn(512, k) * .05
    V0 = torch.randn(1000, k) * .1
    b0 = torch.zeros(k)
    box_run(f"rand{seed}", M0, V0, b0, ybar.clone())

best_train = min(results["runs"][r]["train"] for r in results["runs"] if r != "read")
best_sealed = min(results["runs"][r]["sealed"] for r in results["runs"] if r != "read")
results["gain_train"] = 1 - best_train / read_train
results["gain_sealed"] = 1 - best_sealed / read_sealed
print("gain train", results["gain_train"], "gain sealed", results["gain_sealed"], flush=True)
name = "s5_gates_smoke.json" if a.smoke else f"s5_gates_k{k}_n{n}.json"
with open(os.path.join(OUT, name), "w") as f:
    json.dump(results, f, indent=1)
