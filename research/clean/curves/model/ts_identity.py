# serves: TQ12
"""Check of Lemma TQ12 (teacher-student identity), and the realism check of Hypothesis H-stat on a net whose whole P we hold.

Part A (exact, no training): random heads W, W' on random features with any labels: the identity of TQ12 part 2, with the
stationarity errors G and g written out. Fails if the two sides differ by more than 1e-9.
Part B (realism): a small residual convolutional net on all 50000 CIFAR-100 training photos, no augmentation, SGD with momentum and
weight decay lambda, lr decayed to near 0. P = those 50000 photos (uniform). Measured at the end, on P itself:
  eps_stat = |G| / |lambda W|, cos(M, lambda W) with M = E[(e - s) h^T], |g|_1; train and test accuracy.
  Then the identity on the trained head, with W' = the best rank-k SVD-truncated head, as a float-error check.
  Then a head polished to the exact optimum of R on the fixed features (L-BFGS): eps_stat there is near 0 by construction.
Prediction, written before the run: eps_stat below 1 at the end of SGD with a decayed lr; identity to float error in every case.
Usage: ts_identity.py [--smoke] [--epochs N]"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--wd", type=float, default=5e-4)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(1101)


def identity_terms(W, b, W2, b2, h, lab, lam):
    """All in float64. Returns (left side, right side, D, G, g)."""
    l1, l2 = h @ W.T + b, h @ W2.T + b2
    s = torch.softmax(l1, 1)
    ls = torch.log_softmax(l1, 1)
    lq = torch.log_softmax(l2, 1)
    D = (s * (ls - lq)).sum(1).mean()
    e = F.one_hot(lab, s.shape[1]).to(s.dtype)
    M = (e - s).T @ h / h.shape[0]
    G = M - lam * W
    g = (e - s).mean(0)
    ce1 = -ls[torch.arange(len(lab)), lab].mean()
    ce2 = -lq[torch.arange(len(lab)), lab].mean()
    left = (ce2 + lam / 2 * (W2 ** 2).sum()) - (ce1 + lam / 2 * (W ** 2).sum())
    right = D + lam / 2 * ((W2 - W) ** 2).sum() - (G * (W2 - W)).sum() - (g * (b2 - b)).sum()
    return float(left), float(right), float(D), G, g, M


fails = 0
worst = 0.0
for trial in range(50):
    n, d, C = 200, 8, 5
    h = torch.randn(n, d, dtype=torch.float64)
    lab = torch.randint(0, C, (n,))
    W, W2 = torch.randn(C, d, dtype=torch.float64), torch.randn(C, d, dtype=torch.float64)
    b, b2 = torch.randn(C, dtype=torch.float64), torch.randn(C, dtype=torch.float64)
    lam = float(torch.rand(1)) * .1
    l, r, *_ = identity_terms(W, b, W2, b2, h, lab, lam)
    worst = max(worst, abs(l - r))
    fails += abs(l - r) > 1e-9
print(f"part A: 50 random heads, worst |left - right| = {worst:.2e}, fails {fails}", flush=True)
out = dict(partA_worst=worst, partA_fails=int(fails))

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
d = np.load(os.path.join(HERE, "..", "..", "hf", "cifar100", "cifar100.npz"))
keys = {k: d[k] for k in d.files}
Xtr = [v for k, v in keys.items() if "train" in k and v.ndim == 4][0]
Ytr = [v for k, v in keys.items() if "train" in k and v.ndim == 1][0]
Xte = [v for k, v in keys.items() if "test" in k and v.ndim == 4][0]
Yte = [v for k, v in keys.items() if "test" in k and v.ndim == 1][0]


def to_t(X):
    t = torch.from_numpy(np.array(X)).float() / 255.0
    if t.shape[-1] == 3:
        t = t.permute(0, 3, 1, 2).contiguous().clone()
    return (t - .5) / .25


Xtr, Xte = to_t(Xtr), to_t(Xte)
Ytr, Yte = torch.from_numpy(np.asarray(Ytr)).long(), torch.from_numpy(np.asarray(Yte)).long()
if a.smoke:
    Xtr, Ytr, Xte, Yte = Xtr[:2000], Ytr[:2000], Xte[:500], Yte[:500]
    a.epochs = 2


class Block(nn.Module):
    def __init__(s, i, o, st):
        super().__init__()
        s.c1, s.b1 = nn.Conv2d(i, o, 3, st, 1, bias=False), nn.BatchNorm2d(o)
        s.c2, s.b2 = nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o)
        s.sk = nn.Identity() if i == o and st == 1 else nn.Sequential(nn.Conv2d(i, o, 1, st, bias=False), nn.BatchNorm2d(o))

    def forward(s, x):
        return F.relu(s.b2(s.c2(F.relu(s.b1(s.c1(x))))) + s.sk(x))


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.c0, s.b0 = nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32)
        s.body = nn.Sequential(Block(32, 32, 1), Block(32, 64, 2), Block(64, 64, 1), Block(64, 128, 2), Block(128, 128, 1),
                               Block(128, 256, 2))
        s.fc = nn.Linear(256, 100)

    def feat(s, x):
        return s.body(F.relu(s.b0(s.c0(x)))).mean(3).mean(2)

    def forward(s, x):
        return s.fc(s.feat(x))


m = Net().to(DEV)
bs = 256
steps = a.epochs * (len(Xtr) // bs)
opt = torch.optim.SGD(m.parameters(), lr=.1, momentum=.9, weight_decay=a.wd, nesterov=True)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
t0 = time.time()
for ep in range(a.epochs):
    m.train()
    perm = torch.randperm(len(Xtr))
    for b0 in range(0, len(Xtr) - bs + 1, bs):
        idx = perm[b0:b0 + bs]
        loss = F.cross_entropy(m(Xtr[idx].to(DEV)), Ytr[idx].to(DEV))
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    print(f"epoch {ep} loss {float(loss):.3f} sec {time.time() - t0:.0f}", flush=True)
m.eval()


def feats(X):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 1000):
            out.append(m.feat(X[i:i + 1000].to(DEV)).cpu())
    return torch.cat(out).double()


h, hte = feats(Xtr), feats(Xte)
W = m.fc.weight.detach().cpu().double()
b = m.fc.bias.detach().cpu().double()
lam = a.wd
_, _, _, G, g, M = identity_terms(W, b, W, b, h, Ytr, lam)
acc = float(((h @ W.T + b).argmax(1) == Ytr).double().mean())
acc_te = float(((hte @ W.T + b).argmax(1) == Yte).double().mean())
row = dict(lam=lam, train_acc=acc, test_acc=acc_te, eps_stat=float(G.norm() / (lam * W.norm())),
           cos_M_lamW=float((M * W).sum() / (M.norm() * W.norm())), g_l1=float(g.abs().sum()),
           mean_conf=float(torch.softmax(h @ W.T + b, 1).max(1).values.mean()))
print(row, flush=True)
out["trained"] = row
U, Sv, Vt = torch.linalg.svd(W, full_matrices=False)
for k in (5, 10, 20, 40):
    W2 = (U[:, :k] * Sv[:k]) @ Vt[:k]
    l, r, D, *_ = identity_terms(W, b, W2, b, h, Ytr, lam)
    print(f"k={k}: SVD-truncated head: identity left {l:.6f} right {r:.6f}, D {D:.4f}", flush=True)
    out[f"svd_{k}"] = dict(left=l, right=r, D=D)
# polish the head to the exact optimum of R on the fixed features
W3 = W.clone().requires_grad_(True)
b3 = b.clone().requires_grad_(True)
opt2 = torch.optim.LBFGS([W3, b3], lr=1, max_iter=500, history_size=30, line_search_fn="strong_wolfe", tolerance_grad=1e-9,
                         tolerance_change=1e-14)


def cl():
    opt2.zero_grad()
    v = F.cross_entropy(h @ W3.T + b3, Ytr) + lam / 2 * (W3 ** 2).sum()
    v.backward()
    return v


opt2.step(cl)
W3, b3 = W3.detach(), b3.detach()
_, _, _, G3, g3, M3 = identity_terms(W3, b3, W3, b3, h, Ytr, lam)
pol = dict(eps_stat=float(G3.norm() / (lam * W3.norm())), g_l1=float(g3.abs().sum()),
           train_acc=float(((h @ W3.T + b3).argmax(1) == Ytr).double().mean()),
           test_acc=float(((hte @ W3.T + b3).argmax(1) == Yte).double().mean()),
           dW=float((W3 - W).norm() / W.norm()))
print("polished head", pol, flush=True)
out["polished"] = pol
if not a.smoke:
    json.dump(out, open(os.path.join(HERE, "ts_identity.json"), "w"), indent=1)
    np.savez(os.path.join(HERE, "..", "cache", "map", "cifar_net.npz"), h=h.float().numpy(), hte=hte.float().numpy(),
             lab=Ytr.numpy(), labte=Yte.numpy(), W=W.numpy(), b=b.numpy(), W3=W3.numpy(), b3=b3.numpy(), lam=lam)
print("done", flush=True)
