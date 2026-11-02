# serves: TQ31
"""Card 3-37 line I3, for the network of TQ31: does a real training run reach a small instance from a random start?

Estimator, fixed before the run. n = 5 and 6 graphs of tq31_anytail.py's form, a = 1, rho = .31. One hidden layer of 2 n ReLU units, all weights trained, inputs as in Definition DQ18:
two coordinates per area, h at 1, s at 1 or 3, others 0. Random init, 5 seeds. Objective: cross-entropy on labels. Two label forms:
  soft: the labels are the table itself. A global minimum of this objective is N, for every table, so I3's second clause holds trivially. Said plainly on card 3-37.
  sampled: 2000 labels per point drawn from the table, cross-entropy on those counts.
Adam, 30000 steps, full batch, lr 3e-3 with cosine decay.
Report: KL from the table to the trained net's outputs. Then the DQ18 claims the trained net could meet: its units are not aligned with areas in general, so I report the share of each unit's squared
activation that falls in its best area, as a plain number, in place of a claim's error.

Line, fixed before the run: "reached" if soft KL is under 1e-3, against V_b(1) of about 1e-2 on the same table, so the trained net is close on the scale that matters.
PREDICTION, written before the first run (2026-11-02): soft KL under 1e-3 on most seeds. Units are not aligned with areas: mean best-area share under .7. So the outputs are reached and the map claims are not at 0.

Usage: tq31_trained.py [--smoke]
"""
import argparse
import itertools

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
STEPS = 300 if args.smoke else 30000
rng = np.random.default_rng(11)
for n in (5, 6):
    pairs = list(itertools.combinations(range(n), 2))
    edges = [p for p in pairs if rng.random() < .6]
    a, rho = 1.0, .31
    D = float(np.exp(np.ceil(np.log(n ** 2 * 50))))
    M = np.zeros((n, n))
    W = np.ones((n, n))
    deg = np.zeros(n)
    for u, v in edges:
        M[u, v] = M[v, u] = -1
        W[u, v] = W[v, u] = D
        deg[u] += 1
        deg[v] += 1
    for v in range(n):
        M[v, v] = 1
        W[v, v] = D * max(deg[v], 1)
    rs = W.sum(1)
    Fill = np.ones((n, n)) + np.diag(rs.max() - rs)
    Wa = np.concatenate([W, Fill], 1)
    Ma = np.concatenate([M, np.zeros((n, n))], 1)
    S_ = rs.max() + n
    B = float(np.log(S_ * (1 - rho) / rho / (1 + 1 / np.e)))
    C = 1 + n + 2 * n
    y0 = torch.zeros(n, C)
    y0[:, 0] = B - 1
    y0[torch.arange(n), 1 + torch.arange(n)] = B
    y0[:, 1 + n:] = torch.from_numpy(np.log(Wa))
    mv = torch.zeros_like(y0)
    mv[:, 1 + n:] = torch.from_numpy(Ma) * a / 2
    Y = torch.cat([y0 - mv, y0 + mv])
    T = torch.softmax(Y, 1)
    X = torch.zeros(2 * n, 2 * n)
    for w in range(n):
        for j, val in enumerate((1., 3.)):
            X[w + j * n, w] = 1.
            X[w + j * n, n + w] = val
    area = torch.cat([torch.arange(n), torch.arange(n)])
    for form in ("soft", "sampled"):
        if form == "sampled":
            g = torch.Generator().manual_seed(3)
            Tgt = torch.stack([torch.multinomial(t, 2000, replacement=True, generator=g).bincount(minlength=C).double() / 2000 for t in T])
        else:
            Tgt = T
        res = []
        for seed in range(2 if args.smoke else 5):
            torch.manual_seed(seed)
            net = torch.nn.Sequential(torch.nn.Linear(2 * n, 2 * n), torch.nn.ReLU(), torch.nn.Linear(2 * n, C))
            o = torch.optim.Adam(net.parameters(), lr=3e-3)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(o, STEPS)
            for st in range(STEPS):
                o.zero_grad()
                l = -(Tgt * torch.log_softmax(net(X), 1)).sum(1).mean()
                l.backward()
                o.step()
                sch.step()
            with torch.no_grad():
                kl = float((T * (T.clamp_min(1e-300).log() - torch.log_softmax(net(X), 1))).sum(1).mean())
                h = net[1](net[0](X))
                by_area = torch.stack([(h[area == w] ** 2).sum(0) for w in range(n)])
                share = float((by_area.max(0).values / by_area.sum(0).clamp_min(1e-12)).mean())
            res.append((kl, share))
        print(f"n={n} U-graph edges={len(edges)} {form}: KL per seed {[round(r[0], 5) for r in res]}, best-area share per seed {[round(r[1], 2) for r in res]}", flush=True)
