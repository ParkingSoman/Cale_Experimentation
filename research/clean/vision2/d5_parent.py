"""D5 does each block-17 neuron's trained vote change track a specific block-16 neuron (its parent)? For the 300 most
important block-17 neurons: dz_j = trained vote - original vote (block 16 dropped), on validation positions.
  - best block-16 neuron by |corr(dz_j, h_i)|; is it among j's top-5 edge parents in block 16 (e1_edges)?
  - R^2 of dz_j from its top-5 EDGE parents' activations (no search over 3072) vs from 5 random block-16 neurons
  - coefficient beta (dz_j ~ beta h_i) for the top edge parent vs the edge value: ratio across neurons (a constant?)"""
import os, sys, copy
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); gelu = nn.gelu
rk = np.zeros(len(z["ana"]), int)
for c in range(1000): ix = np.where(z["yA"] == c)[0]; rk[ix] = np.arange(len(ix))
VA = np.where(rk >= 18)[0][:1000]; b16, b17 = m.blocks[16], m.blocks[17]; t17 = copy.deepcopy(b17); t17.load_weights([(k, mx.array(v)) for k, v in np.load(os.path.join(D, sys.argv[1])).items()])
E = np.load(os.path.join(HF, "imagenet", "e1_edges.npz"))["17"]; OFF16 = 384 * 3 + 768 * 3 + 1536 * 9 + 3072; E16 = E[:, OFF16:OFF16 + 3072]   # (3072 children, 3072 block-16 parents)
dZ, Hp, Zo, Lost = [], [], [], []
for a in range(0, len(VA), 100):
    x = mx.array(z["XA"][VA[a:a + 100]]).astype(mx.float32)
    zo = b17.fc1(b17.pre(x)); dZ.append(np.array(t17.fc1(t17.pre(x)) - zo).reshape(-1, 3072)); Hp.append(np.array(gelu(b16.fc1(b16.pre(x)))).reshape(-1, 3072))
    Lost.append(np.array(b17.fc1(b17.pre(b16(x))) - zo).reshape(-1, 3072))
dZ, Hp, Lost = (np.concatenate(v).astype(np.float64) for v in (dZ, Hp, Lost))
imp = np.linalg.norm(np.array(b17.fc2.weight) * np.array(b17.gamma)[:, None], axis=0) * dZ.std(0); J = np.argsort(-imp)[:300]
zs = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-9); Hz = zs(Hp)
def r2(Y, F):
    A = np.c_[F, np.ones(len(F))]; h = np.repeat(np.arange(len(A) // 49) % 2 == 0, 49); w = np.linalg.lstsq(A[h], Y[h], rcond=None)[0]; return 1 - ((A[~h] @ w - Y[~h]) ** 2).mean() / Y[~h].var()
mc = []; rng = np.random.default_rng(0); hit, r_edge, r_rand, r_lost, ratio = 0, [], [], [], []
for j in J:
    c = Hz.T @ zs(dZ[:, j:j + 1])[:, 0] / len(Hz); best = int(np.argmax(np.abs(c))); top5 = np.argsort(-np.abs(E16[j]))[:5]; hit += best in top5; mc.append(abs(c[best]))
    r_edge.append(r2(dZ[:, j], Hp[:, top5])); r_rand.append(r2(dZ[:, j], Hp[:, rng.choice(3072, 5, replace=False)])); r_lost.append(r2(dZ[:, j], Lost[:, j:j + 1]))
    p = top5[0]; hc = Hp[:, p] - Hp[:, p].mean(); beta = (hc @ (dZ[:, j] - dZ[:, j].mean())) / (hc @ hc + 1e-12); ratio.append(beta / (E16[j, p] + 1e-12))
print(f"300 most important block-17 neurons (trained run {sys.argv[1]}):")
print(f"  best-correlated block-16 neuron is among the top-5 EDGE parents: {hit}/300 (chance ~{300 * 5 / 3072:.1f})")
print(f"  R^2 of the vote change from: top-5 edge parents' activations {np.median(r_edge):.3f} | 5 random block-16 neurons {np.median(r_rand):.3f} | the lost parent vote itself {np.median(r_lost):.3f}")
ratio = np.array(ratio); print(f"  max |corr| with any block-16 neuron: median {np.median(mc):.3f}");print(f"  beta / edge for the top edge parent: median {np.median(ratio):.3f}, IQR {np.percentile(ratio, 25):.3f} to {np.percentile(ratio, 75):.3f}")
