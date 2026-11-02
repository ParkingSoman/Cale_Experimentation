"""Family crush: replace one 3b family (e.g. the 10 curve neurons) by k coarse neurons. Everything else is intact.

This is the exact smaller network: 3b has 480 - n + k channels. A coarse neuron reads the same input as the family
(the 5x5 bottleneck, 32 x 5 x 5) and has a ReLU. Layer 4a gets k new input columns (one per coarse neuron, in each of
its four input convs), after the same max-pool. Only the coarse reads, biases and 4a columns are trained.
Loss: KL(intact || crushed) on logits, cached intact logits (with mirror flips). Early stop on dev KL.
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn
from inception import *

IN4A = ["mixed4a_1x1_pre_relu_conv", "mixed4a_3x3_bottleneck_pre_relu_conv",
        "mixed4a_5x5_bottleneck_pre_relu_conv", "mixed4a_pool_reduce_pre_relu_conv"]


def pool3(x):
    return F.max_pool2d(F.pad(x, (0, 1, 0, 1), value=float('-inf')), 3, 2)


class Family(nn.Module):
    """k coarse neurons replacing family `fam` (indices into the 5x5 branch of 3b)."""
    def __init__(self, m, fam, k, init="random", seed=0, ks=5):
        super().__init__()
        self.ks = ks
        g = torch.Generator().manual_seed(seed)
        self.fam = list(fam); self.k = k
        self.keep = [i for i in range(480) if i not in self.fam]
        W5 = m.mixed3b_5x5_pre_relu_conv.weight.detach()[[i - 320 for i in self.fam]].cpu()   # (n, 32, 5, 5)
        b5 = m.mixed3b_5x5_pre_relu_conv.bias.detach()[[i - 320 for i in self.fam]].cpu()
        # parameter units: every entry is O(1), so Adam moves all of them at the same relative rate
        self.wscale = float(W5.flatten(1).norm(dim=1).mean()) / np.sqrt(800)
        self.bscale = 500.0   # typical std of a curve neuron's pre-activation (S1: 460-580)
        cols = [getattr(m, n).weight.detach()[:, self.fam, 0, 0].cpu() for n in IN4A]          # each (out, n)
        self.cscale = [float(c.norm(dim=0).mean()) / np.sqrt(c.shape[0]) for c in cols]
        if init == "random":
            self.read = nn.Parameter(torch.randn(k, 32, ks, ks, generator=g))
            self.bias = nn.Parameter(torch.zeros(k))
            self.cols = nn.ParameterList([nn.Parameter(torch.randn(c.shape[0], k, generator=g)) for c in cols])
        elif init == "copy":   # exact original when k == n
            assert k == len(self.fam)
            self.read = nn.Parameter(W5 / self.wscale); self.bias = nn.Parameter(b5 / self.bscale)
            self.cols = nn.ParameterList([nn.Parameter(c / s) for c, s in zip(cols, self.cscale)])
        elif init.startswith("member:"):   # start every coarse neuron as one real family member (k members listed)
            mem = [self.fam.index(int(t)) for t in init.split(":")[1].split(",")]; assert len(mem) == k
            self.read = nn.Parameter(W5[mem] / self.wscale); self.bias = nn.Parameter(b5[mem] / self.bscale)
            self.cols = nn.ParameterList([nn.Parameter(c[:, mem] / s) for c, s in zip(cols, self.cscale)])
        else:
            raise ValueError(init)

    def calibrate(self, ins, rate=0.12):
        """Set each bias so the neuron is on at `rate` of positions on this batch (keeps a random neuron alive)."""
        with torch.no_grad():
            p = self.read.shape[-1] // 2
            z = F.conv2d(F.pad(ins[2], (p, p, p, p)), self.read * self.wscale)
            q = torch.quantile(z.transpose(0, 1).reshape(self.k, -1)[:, ::7].float(), 1 - rate, dim=1)
            self.bias.copy_(-q / self.bscale)

    def set_weights(self, read, bias, cols):
        """Install weights given in original units (read (k,32,5,5), bias (k,), cols list of (out,k))."""
        with torch.no_grad():
            self.read.copy_(read / self.wscale); self.bias.copy_(bias / self.bscale)
            for p, c, s in zip(self.cols, cols, self.cscale): p.copy_(c / s)

    def weights(self):
        return (self.read.detach() * self.wscale, self.bias.detach() * self.bscale,
                [p.detach() * s for p, s in zip(self.cols, self.cscale)])

    def pre(self, ins):
        """Coarse pre-activations (B, k, 28, 28) from the four 3b inputs."""
        p = self.read.shape[-1] // 2
        return F.conv2d(F.pad(ins[2], (p, p, p, p)), self.read * self.wscale, self.bias * self.bscale)


def forward_family(m, fm, ins):
    """Logits of the crushed network."""
    h = F.relu(pre_3b(m, ins[0]))[:, fm.keep]                     # other 470 channels, intact
    c = F.relu(fm.pre(ins))                                        # k coarse channels
    x = torch.cat([pool3(h), pool3(c)], 1)
    p = F.max_pool2d(F.pad(x, (1, 1, 1, 1), value=float('-inf')), 3, 1)
    outs = []
    for name, col, s, inp in zip(IN4A, fm.cols, fm.cscale, [x, x, x, p]):
        conv = getattr(m, name); W = torch.cat([conv.weight[:, fm.keep, 0, 0], col * s], 1)[:, :, None, None]
        cb = getattr(fm, "colbias", None)
        outs.append(F.conv2d(inp, W, conv.bias if cb is None else conv.bias + cb[len(outs)]))
    a = m.mixed4a_1x1(outs[0]); b = m.mixed4a_3x3_bottleneck(outs[1]); cc = m.mixed4a_5x5_bottleneck(outs[2])
    d = m.mixed4a_pool_reduce(outs[3])
    b = m.mixed4a_3x3(m.mixed4a_3x3_pre_relu_conv(F.pad(b, (1, 1, 1, 1))))
    cc = m.mixed4a_5x5(m.mixed4a_5x5_pre_relu_conv(F.pad(cc, (2, 2, 2, 2))))
    h = torch.cat([a, b, cc, d], 1)
    for n in ["mixed4b", "mixed4c", "mixed4d", "mixed4e"]:
        h = mixed(m, n, h)
    h = pool3(h)
    for n in ["mixed5a", "mixed5b"]:
        h = mixed(m, n, h)
    return m.softmax2_pre_activation_matmul(F.avg_pool2d(h, 7).reshape(-1, 1024))


def kl(t, s):
    lt = F.log_softmax(t, 1); ls = F.log_softmax(s, 1)
    return (lt.exp() * (lt - ls)).sum(1)


class Data:
    def __init__(self, seed=0, ndev=0):
        self.X, self.y = load_images(); self.tr, self.dv, self.se = split(self.y)
        self.T = np.load(os.path.join(HERE, "cache", "teacher_logits.npy"), mmap_mode="r")
        self.TF = np.load(os.path.join(HERE, "cache", "teacher_logits_flip.npy"), mmap_mode="r")
        self.lab = np.load(os.path.join(HERE, "label_map.npy"))
        if ndev: self.dv = np.sort(np.random.default_rng(seed).choice(self.dv, ndev, replace=False))


def evaluate(m, fm, D, idx, bs=100):
    ks, tp = [], []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]; lg = forward_family(m, fm, inputs_3b(m, to_3a(m, prep(D.X[b]))))
            t = torch.from_numpy(D.T[b].astype(np.float32)).to(DEV)
            ks.append(kl(t, lg).cpu().numpy()); tp.append(lg[:, D.lab].argmax(1).cpu().numpy() == D.y[b])
    k = np.concatenate(ks)
    return float(k.mean()), float(k.std() / np.sqrt(len(k))), float(np.concatenate(tp).mean())


def train(m, fm, D, lr=1e-2, steps=2000, bs=64, seed=0, eval_every=200, patience=5, log=print):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(fm.parameters(), lr=lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best = (1e9, None, 0); hist = []; run = []; bad = 0; t0 = time.time()
    for step in range(1, steps + 1):
        b = np.sort(rng.choice(D.tr, bs, replace=False)); flip = rng.random(bs) < 0.5
        xb = np.asarray(D.X[b]); xb = np.where(flip[:, None, None, None], xb[:, :, ::-1], xb)
        with torch.no_grad():
            ins = inputs_3b(m, to_3a(m, prep(xb)))
            t = torch.from_numpy(np.where(flip[:, None], D.TF[b], D.T[b]).astype(np.float32)).to(DEV)
        loss = kl(t, forward_family(m, fm, ins)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); run.append(loss.item())
        if step % eval_every == 0 or step == steps:
            dkl, dse, dtop = evaluate(m, fm, D, D.dv)
            hist.append(dict(step=step, train_kl=float(np.mean(run)), dev_kl=dkl, dev_se=dse, dev_top1=dtop, sec=round(time.time() - t0)))
            run = []; log(json.dumps(hist[-1]))
            if dkl < best[0]:
                best = (dkl, {k: v.detach().clone() for k, v in fm.state_dict().items()}, step); bad = 0
            else:
                bad += 1
                if bad >= patience: break
    fm.load_state_dict(best[1])
    return best[0], best[2], hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1); ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="random"); ap.add_argument("--ndev", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=200); ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--fam", default="curves"); ap.add_argument("--tag", default=None); ap.add_argument("--ks", type=int, default=5)
    a = ap.parse_args()
    fam = {"curves": CURVES}[a.fam]
    tag = a.tag or f"fam-{a.fam}-k{a.k}-lr{a.lr:g}-s{a.seed}-{a.init}"
    torch.manual_seed(a.seed)
    m = load_model(); D = Data(seed=0, ndev=a.ndev)
    for p in m.parameters(): p.requires_grad_(False)
    fm = Family(m, fam, a.k, init=a.init, seed=a.seed, ks=a.ks).to(DEV)
    if a.init == "random":
        with torch.no_grad(): fm.calibrate(inputs_3b(m, to_3a(m, prep(D.X[np.sort(D.tr[:64])]))))
    if a.init == "copy": print("copy init (should be 0):", evaluate(m, fm, D, D.dv[:300]))
    best, step, hist = train(m, fm, D, lr=a.lr, steps=a.steps, seed=a.seed, eval_every=a.eval_every, patience=a.patience)
    os.makedirs(os.path.join(HERE, "cache", "ckpt"), exist_ok=True)
    torch.save(dict(state=fm.state_dict(), fam=fam, k=a.k, ks=a.ks, args=vars(a)), os.path.join(HERE, "cache", "ckpt", tag + ".pt"))
    json.dump(dict(tag=tag, args=vars(a), best_dev_kl=best, best_step=step, hist=hist),
              open(os.path.join(HERE, "..", "..", "..", "results", f"curves-{tag}.json"), "w"), indent=1)
    print("BEST", tag, best, "at step", step, flush=True)


if __name__ == "__main__":
    main()
