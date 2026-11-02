"""BEAT I5: children weights of each unit of the k = 3 run started from members 340/343/406 (50/170/290 deg)."""
import numpy as np, torch
from inception import *
from family import pool3
from w_fisher import load_build
m = load_model(); X, y = load_images(); tr, dv, se = split(y); fam = list(np.load("cache/s1_fine.npz")["fam"])
ANG = [10, 50, 70, 120, 170, 180, 250, 290, 320, 350]
flat = lambda t: t[:, :, 2:12, 2:12].permute(1, 0, 2, 3).reshape(t.shape[1], -1).cpu().double().numpy()
ix = np.sort(np.random.default_rng(41).permutation(tr)[:1500])[:500]; fm = load_build(m, fam, "I5-k3-m50-170-290")
Z, C = [], []
with torch.no_grad():
    for i in range(0, len(ix), 50):
        ins = inputs_3b(m, to_3a(m, prep(X[ix[i:i + 50]]))); Z.append(flat(pool3(F.relu(pre_3b(m, ins[0])[:, fam])))); C.append(flat(pool3(F.relu(fm.pre(ins)))))
Z, C = np.concatenate(Z, 1), np.concatenate(C, 1); D = np.concatenate([Z.T, np.ones((Z.shape[1], 1))], 1)
np.set_printoptions(precision=3, suppress=True, linewidth=160)
for j, st in enumerate([1, 4, 7]):
    w = np.linalg.lstsq(D, C[j], rcond=None)[0]; r2 = 1 - ((C[j] - D @ w) ** 2).mean() / C[j].var(); l = w[:10] / np.abs(w[:10]).sum()
    print(f"unit {j} start {ANG[st]} deg: R^2 {r2:.2f}, on {(C[j] > 0).mean():.2f}, argmax member {ANG[int(np.argmax(l))]} deg | lam", dict(zip(ANG, np.round(l, 3))))
