"""Distilled pair: a narrow copy of Base's stage-4 block-0 MLP, trained to reproduce that MLP's output.

Base (convnext_base) is frozen. Block 0 of stage 4 computes y = layer_scale * W2 GELU(W1 u + b1) + ... on
u = LayerNorm(dwconv(stream)) (1,024 per position), 4,096 hidden neurons. A coarse MLP with k hidden neurons
(random start) is trained on the same u to reproduce the same y (squared error). So each coarse neuron reads exactly
Base's input, and its relation to Base's 4,096 neurons can be read off with no upstream drift.

  --cache      one Base pass: u and y for 6,000 fit photos and 1,500 test photos (fp16, ~1.5 GB in results/pair/distill/)
  --k K --seed S   train, save weights; report test fit, and top-1 / KL of Base with the coarse MLP plugged in.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from cache import DEV, OUT, load, net, split

D = os.path.join(OUT, "distill")
os.makedirs(D, exist_ok=True)


def photo_sets(ph):
    fit = split(ph.lab)
    tr, te = [], []
    for c in range(1000):
        tr += np.where((ph.lab == c) & fit)[0][:6].tolist()
        te += np.where((ph.lab == c) & ~fit)[0][:2].tolist()
    rng = np.random.default_rng(0)
    te = rng.permutation(te)[:1500]
    return np.array(sorted(tr)), np.array(sorted(te))


def cache():
    torch.set_grad_enabled(False)
    ph = net.Photos("read")
    tr, te = photo_sets(ph)
    m = load("base")
    blk = m.features[7][0]
    H = {}
    blk.block[2].register_forward_hook(lambda mod, i, o: H.__setitem__("u", o))
    blk.block[5].register_forward_hook(lambda mod, i, o: H.__setitem__("y", o))
    ls = blk.layer_scale.detach().view(1, -1)
    for name, ix in (("train", tr), ("test", te)):
        U = np.zeros((len(ix) * 49, 1024), np.float16)
        Y = np.zeros((len(ix) * 49, 1024), np.float16)
        t0 = time.time()
        for k in range(0, len(ix), 50):
            ii = ix[k:k + 50]
            m(ph.tensor(ii).to(DEV))
            U[k * 49:(k + len(ii)) * 49] = H["u"].reshape(-1, 1024).float().cpu().numpy()
            Y[k * 49:(k + len(ii)) * 49] = (H["y"].reshape(-1, 1024).float() * ls).cpu().numpy()
            if (k // 50) % 30 == 0:
                print(f"  {name} {k + len(ii)}/{len(ix)} {time.time() - t0:.0f}s", flush=True)
        np.save(os.path.join(D, f"U_{name}.npy"), U)
        np.save(os.path.join(D, f"Y_{name}.npy"), Y)
        np.save(os.path.join(D, f"photos_{name}.npy"), ix)


class MLP(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.l1 = torch.nn.Linear(1024, k)
        self.l2 = torch.nn.Linear(k, 1024)

    def forward(self, u):
        return self.l2(F.gelu(self.l1(u)))


def train(k, seed, epochs, batch=4096, lr=2e-3):
    torch.set_grad_enabled(True)  # cache.py turns gradients off on import
    torch.manual_seed(seed)
    U = torch.from_numpy(np.load(os.path.join(D, "U_train.npy")))
    Y = torch.from_numpy(np.load(os.path.join(D, "Y_train.npy")))
    Ut = torch.from_numpy(np.load(os.path.join(D, "U_test.npy"))).float().to(DEV)
    Yt = torch.from_numpy(np.load(os.path.join(D, "Y_test.npy"))).float().to(DEV)
    net_ = MLP(k).to(DEV)
    opt = torch.optim.AdamW(net_.parameters(), lr=lr, weight_decay=0.0)
    steps = epochs * (len(U) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    yvar = Yt.var(0).sum()
    t0 = time.time()
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(len(U), generator=g)
        for i in range(len(U) // batch):
            j = perm[i * batch:(i + 1) * batch]
            u, y = U[j].float().to(DEV), Y[j].float().to(DEV)
            loss = (net_(u) - y).pow(2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            e = sum(((net_(Ut[a:a + 20000]) - Yt[a:a + 20000]).pow(2).sum()) for a in range(0, len(Ut), 20000))
            r2 = 1 - e / len(Ut) / yvar
        print(f"  k{k} s{seed} epoch {ep + 1}/{epochs}  test R2 {float(r2):.4f}  {time.time() - t0:.0f}s", flush=True)
    torch.save(net_.state_dict(), os.path.join(D, f"mlp_k{k}_s{seed}.pt"))
    return net_, float(r2)


def plug_eval(net_, k, seed):
    """Top-1 and KL of Base with block 0's MLP replaced, on the test photos."""
    torch.set_grad_enabled(False)
    ph = net.Photos("read")
    te = np.load(os.path.join(D, "photos_test.npy"))
    m = load("base")
    blk = m.features[7][0]
    state = {"on": False}

    def hook(mod, inp, out):
        if not state["on"]:
            return out
        u = blk.block[2](blk.block[1](blk.block[0](inp[0])))
        y = net_(u) / blk.layer_scale.view(1, 1, 1, -1)
        return y
    blk.block[5].register_forward_hook(lambda mod, i, o: o)  # keep structure
    orig = blk.forward

    def fwd(x):
        if not state["on"]:
            return orig(x)
        u = blk.block[2](blk.block[1](blk.block[0](x)))
        y = net_(u).permute(0, 3, 1, 2)
        return x + y  # stochastic depth is identity in eval
    blk.forward = fwd
    kl, c0, c1, n = 0.0, 0, 0, 0
    for a in range(0, len(te), 50):
        ii = te[a:a + 50]
        x = ph.tensor(ii).to(DEV)
        lab = torch.tensor(ph.lab[ii], device=DEV)
        state["on"] = False
        y0 = m(x)
        state["on"] = True
        y1 = m(x)
        kl += float((F.softmax(y0, 1) * (F.log_softmax(y0, 1) - F.log_softmax(y1, 1))).sum())
        c0 += int((y0.argmax(1) == lab).sum())
        c1 += int((y1.argmax(1) == lab).sum())
        n += len(ii)
    return dict(kl=kl / n, top1_base=c0 / n, top1_plugged=c1 / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()
    if a.cache:
        cache()
        return
    net_, r2 = train(a.k, a.seed, a.epochs)
    net_.eval()
    ev = plug_eval(net_, a.k, a.seed)
    ev.update(k=a.k, seed=a.seed, epochs=a.epochs, test_r2=r2)
    print(json.dumps(ev), flush=True)
    json.dump(ev, open(os.path.join(D, f"eval_k{a.k}_s{a.seed}.json"), "w"))


if __name__ == "__main__":
    main()
