"""D2 toy: a known taxonomy, each pattern shown with a random sign (so a linear net cannot solve it) (8 coarse x 8 fine = 64 classes). Train wide (256) and narrow (16) one-hidden-layer GELU nets
(3 seeds each, same data and recipe) and LINEAR versions. Ask what a narrow neuron is in terms of the wide net's
neurons, where the truth (coarse/fine labels) is known. Saves nets to hf/toy/. usage: python d2_toy.py"""
import os, json, time
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from scipy.special import erf
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hf", "toy"); os.makedirs(OUT, exist_ok=True)
gl = lambda z: 0.5 * z * (1 + erf(z / np.sqrt(2)))
D, NC, NF = 64, 8, 8; rng = np.random.default_rng(0)
Ac = rng.normal(0, 1.0, (NC, D)); Bf = rng.normal(0, 0.6, (NC, NF, D))          # coarse template, fine offset
def data(n, r):
    c = r.integers(NC, size=n); f = r.integers(NF, size=n); amp = r.uniform(.5, 1.5, (n, 1))
    sg = r.choice([-1.0, 1.0], size=(n, 1)); x = sg * amp * (Ac[c] + Bf[c, f]) + r.normal(0, 1.0, (n, D)); return x.astype(np.float32), (c * NF + f).astype(np.int32), c
Xtr, ytr, ctr = data(60000, np.random.default_rng(1)); Xte, yte, cte = data(20000, np.random.default_rng(2))
class Net(nn.Module):
    def __init__(s, H, lin=False): super().__init__(); s.l1 = nn.Linear(D, H); s.l2 = nn.Linear(H, NC * NF); s.lin = lin
    def hid(s, x): z = s.l1(x); return z if s.lin else nn.gelu(z)
    def __call__(s, x): return s.l2(s.hid(x))
def train(H, seed, lin=False, steps=3000):
    mx.random.seed(seed); net = Net(H, lin); opt = optim.Adam(1e-3); r = np.random.default_rng(seed)
    lf = lambda m, x, y: nn.losses.cross_entropy(m(x), y).mean(); g = nn.value_and_grad(net, lf)
    for t in range(steps):
        i = r.integers(len(Xtr), size=512); l, gr = g(net, mx.array(Xtr[i]), mx.array(ytr[i])); opt.update(net, gr); mx.eval(net.parameters())
    acc = float((np.array(net(mx.array(Xte))).argmax(1) == yte).mean()); cacc = float((np.array(net(mx.array(Xte))).argmax(1) // NF == cte).mean())
    return net, acc, cacc
if __name__ == "__main__":
    t0 = time.time(); res = {}
    for H in (256, 16):
        for lin in (False, True):
            for seed in (0, 1, 2):
                net, acc, cacc = train(H, seed, lin); k = f"{'lin' if lin else 'gelu'}_{H}_s{seed}"; res[k] = (acc, cacc)
                np.savez(os.path.join(OUT, k + ".npz"), W1=np.array(net.l1.weight), b1=np.array(net.l1.bias), W2=np.array(net.l2.weight), b2=np.array(net.l2.bias))
                print(k, f"fine acc {acc:.3f} coarse acc {cacc:.3f}", f"{time.time() - t0:.0f}s", flush=True)
    np.savez(os.path.join(OUT, "data.npz"), Xte=Xte, yte=yte, cte=cte, Ac=Ac, Bf=Bf)
    json.dump(res, open(os.path.join(OUT, "acc.json"), "w"))
