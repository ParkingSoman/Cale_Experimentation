"""Vision lab (lineage round 3): isotropic ConvNeXt-style net with the essay's clean conditions:
residual stream + LayerNorm + GELU pointwise MLP. Block: x + fc2(gelu(fc1(ln(dwconv7(x))))). Neurons = fc1 units."""
import os, time, json
from functools import partial
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DATA = os.path.join(ROOT, "research", "clean", "hf", "cifar100", "cifar100.npz")
CK = os.path.join(ROOT, "research", "clean", "hf", "vision-ck")
RES = os.path.join(ROOT, "results")
MEAN = np.array([0.507, 0.487, 0.441], np.float32); STD = np.array([0.267, 0.256, 0.276], np.float32)


class Block(nn.Module):
    def __init__(self, C, H):
        super().__init__()
        self.dw = nn.Conv2d(C, C, 7, padding=3, groups=C)
        self.ln = nn.LayerNorm(C)
        self.fc1 = nn.Linear(C, H)
        self.fc2 = nn.Linear(H, C)

    def hidden(self, x):
        return nn.gelu(self.fc1(self.ln(self.dw(x))))

    def __call__(self, x):
        return x + self.fc2(self.hidden(x))


class Net(nn.Module):
    def __init__(self, C, H, D=6, n_cls=100):
        super().__init__()
        self.stem = nn.Conv2d(3, C, 4, stride=4)
        self.ln0 = nn.LayerNorm(C)
        self.blocks = [Block(C, H) for _ in range(D)]
        self.lnf = nn.LayerNorm(C)
        self.head = nn.Linear(C, n_cls)

    def __call__(self, x, upto=None):
        x = self.ln0(self.stem(x))
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x.mean((1, 2))))


def load_data():
    d = np.load(DATA)
    f = lambda x: ((x.astype(np.float32) / 255.0) - MEAN) / STD
    return {"xtr": f(d["x_train"]), "ytr": d["fine_train"].astype(np.int32), "ctr": d["coarse_train"].astype(np.int32),
            "xte": f(d["x_test"]), "yte": d["fine_test"].astype(np.int32), "cte": d["coarse_test"].astype(np.int32)}


def split_test():
    """Fixed split of the 10k test images: 5k analysis (selection, neuron reading), 5k sealed (accuracy only)."""
    p = np.random.default_rng(12345).permutation(10000)
    return np.sort(p[:5000]), np.sort(p[5000:])


def augment(x, rng):
    """Random crop (pad 4, reflect) + horizontal flip, vectorized."""
    n = len(x); xp = np.pad(x, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="reflect")
    i, j = rng.integers(0, 9, n), rng.integers(0, 9, n)
    r = i[:, None] + np.arange(32)[None]; c = j[:, None] + np.arange(32)[None]
    flip = rng.random(n) < .5; c[flip] = c[flip, ::-1]
    return xp[np.arange(n)[:, None, None], r[:, :, None], c[:, None, :]]


def logits(model, x, bs=500):
    return np.concatenate([np.array(model(mx.array(x[i:i + bs]))) for i in range(0, len(x), bs)])


def accuracy(lg, y, c, fine_to_coarse):
    p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    pc = np.zeros((len(p), 20)); np.add.at(pc.T, fine_to_coarse, p.T)
    return {"fine": float((p.argmax(1) == y).mean()), "super": float((pc.argmax(1) == c).mean()),
            "fine_correct": (p.argmax(1) == y), "super_correct": (pc.argmax(1) == c)}


def train(name, C, H, epochs, init_seed=0, data_seed=0, bs=256, lr=2e-3, save_epochs=(1, 2, 4, 8, 16), time_only=False):
    d = load_data(); mx.random.seed(init_seed)
    model = Net(C, H); mx.eval(model.parameters())
    nparam = sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))
    steps = epochs * (50000 // bs)
    sched = optim.cosine_decay(lr, steps)
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.05)
    loss_fn = lambda m, x, y: nn.losses.cross_entropy(m(x), y).mean()
    lg = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        l, g = lg(model, x, y); opt.update(model, g); return l

    rng = np.random.default_rng(data_seed); os.makedirs(CK, exist_ok=True)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        perm = rng.permutation(50000)
        for s in range(50000 // bs):
            idx = perm[s * bs:(s + 1) * bs]
            l = step(mx.array(augment(d["xtr"][idx], rng)), mx.array(d["ytr"][idx])); mx.eval(state)
            if time_only and s == 49:
                return (time.time() - t0) / 50 * (50000 // bs), nparam
        if ep in save_epochs or ep == epochs:
            model.save_weights(os.path.join(CK, f"{name}-ep{ep}.safetensors"))
        print(name, "epoch", ep, "loss %.3f" % float(l), "%.0fs" % (time.time() - t0), flush=True)
    return model, nparam


def load_net(name, C, H, ep):
    m = Net(C, H); m.load_weights(os.path.join(CK, f"{name}-ep{ep}.safetensors")); mx.eval(m.parameters()); return m
