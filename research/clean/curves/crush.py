"""Crushed 3b: k neurons, each reading all four inputs the 3b branches read (3a 1x1, 3x3 bottleneck 3x3,
5x5 bottleneck 5x5, pooled 3a 1x1), ReLU. Reader: linear k -> 480 into the frozen 4a and the rest of the network.
reader='lin+4a' also trains 4a. Loss: KL(intact || crushed) on logits. Early stop on dev KL.

Each original 3b neuron is a special case of a crushed neuron (zeros in the other three inputs),
so fine and coarse neurons live in one weight space: weights_eff() returns them in the original units.
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn
from inception import *

IN_CH = [256, 128, 32, 256]; KS = [1, 3, 5, 1]


class Crush(nn.Module):
    def __init__(self, k, s_in, s_out, s_480):
        super().__init__()
        self.k = k
        self.convs = nn.ModuleList([nn.Conv2d(c, k, ks, padding=ks // 2, bias=False) for c, ks in zip(IN_CH, KS)])
        self.bias = nn.Parameter(torch.zeros(k))
        self.read = nn.Conv2d(k, 480, 1, bias=True)
        self.register_buffer("s_in", torch.tensor(s_in, dtype=torch.float32))   # per input group scale
        self.s_out = float(s_out)   # typical 3b pre-activation std
        nn.init.normal_(self.read.weight, 0, 1.0 / np.sqrt(k)); self.read.weight.data *= s_480 / s_out
        nn.init.zeros_(self.read.bias)

    def pre(self, ins):
        z = sum(cv(x / s) for cv, x, s in zip(self.convs, ins, self.s_in))
        return self.s_out * z + self.bias[None, :, None, None] * self.s_out

    def forward(self, ins):
        c = F.relu(self.pre(ins))
        return self.read(c), c

    def weights_eff(self):
        """Weights on the raw inputs (same units as the original 3b weights) and bias."""
        return [cv.weight.detach() * self.s_out / s for cv, s in zip(self.convs, self.s_in)], self.bias.detach() * self.s_out


def scales(m, X, idx):
    with torch.no_grad():
        ins = inputs_3b(m, to_3a(m, prep(X[idx])))
        s_in = [x.std().item() for x in ins]
        pre = pre_3b(m, ins[0]); s_out = pre.std((0, 2, 3)).mean().item()
        s_480 = F.relu(pre).std().item()
    return s_in, s_out, s_480


def kl(t_logits, s_logits):
    lt = F.log_softmax(t_logits, 1); ls = F.log_softmax(s_logits, 1)
    return (lt.exp() * (lt - ls)).sum(1)


def evaluate(m, cr, X, T, idx, lab, y, bs=100):
    cr.eval(); kls, top = [], []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]; ins = inputs_3b(m, to_3a(m, prep(X[b])))
            lg = from_3b(m, cr(ins)[0]); t = torch.from_numpy(T[b].astype(np.float32)).to(DEV)
            kls.append(kl(t, lg).cpu().numpy()); top.append((lg[:, lab].argmax(1).cpu().numpy() == y[b]))
    cr.train()
    return float(np.concatenate(kls).mean()), float(np.concatenate(top).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--ntrain", type=int, default=0)
    ap.add_argument("--ndev", type=int, default=0); ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=6); ap.add_argument("--reader", default="lin")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or f"k{a.k}-lr{a.lr:g}-s{a.seed}-{a.reader}"
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    m = load_model(); X, y = load_images(); tr, dv, se = split(y)
    T = np.load(os.path.join(HERE, "cache", "teacher_logits.npy"), mmap_mode="r"); TF = np.load(os.path.join(HERE, "cache", "teacher_logits_flip.npy"), mmap_mode="r"); lab = np.load(os.path.join(HERE, "label_map.npy"))
    if a.ntrain: tr = rng.choice(tr, a.ntrain, replace=False)
    if a.ndev: dv = np.sort(rng.choice(dv, a.ndev, replace=False))
    sc = scales(m, X, tr[:64]); cr = Crush(a.k, *sc).to(DEV)
    params = list(cr.parameters())
    PRE = {"lin": (), "lin+4a": ("mixed4a",), "lin+all": ("mixed4", "mixed5", "softmax2")}[a.reader]
    for n, p in m.named_parameters():
        if n.startswith(PRE) and PRE: p.requires_grad_(True); params.append(p)
    opt = torch.optim.Adam(params, lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    best = (1e9, None, 0); hist = []; t0 = time.time(); bad = 0; run = []
    for step in range(1, a.steps + 1):
        b = rng.choice(tr, a.bs, replace=False); xb = np.asarray(X[np.sort(b)]); b = np.sort(b)
        flip = rng.random(a.bs) < 0.5; xb = np.where(flip[:, None, None, None], xb[:, :, ::-1], xb)
        with torch.no_grad():
            ins = inputs_3b(m, to_3a(m, prep(xb)))
            t = torch.from_numpy(np.where(flip[:, None], TF[b], T[b]).astype(np.float32)).to(DEV)   # intact logits, cached
        loss = kl(t, from_3b(m, cr(ins)[0])).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); run.append(loss.item())
        if step % a.eval_every == 0 or step == a.steps:
            dkl, dtop = evaluate(m, cr, X, T, dv, lab, y)
            hist.append(dict(step=step, train_kl=float(np.mean(run)), dev_kl=dkl, dev_top1=dtop, sec=time.time() - t0)); run = []
            print(json.dumps(hist[-1]), flush=True)
            if dkl < best[0]:
                best = (dkl, {k_: v.detach().cpu().clone() for k_, v in cr.state_dict().items()}, step); bad = 0
                if PRE:
                    best[1].update({"net." + n: p.detach().cpu().clone() for n, p in m.named_parameters() if n.startswith(PRE)})
            else:
                bad += 1
                if bad >= a.patience: break
    os.makedirs(os.path.join(HERE, "cache", "ckpt"), exist_ok=True)
    torch.save(dict(state=best[1], k=a.k, args=vars(a), scales=sc), os.path.join(HERE, "cache", "ckpt", tag + ".pt"))
    res = dict(tag=tag, args=vars(a), best_dev_kl=best[0], best_step=best[2], hist=hist)
    json.dump(res, open(os.path.join(HERE, "..", "..", "..", "results", f"curves-{tag}.json"), "w"), indent=1)
    print("BEST", tag, best[0], "at step", best[2])


if __name__ == "__main__":
    main()
