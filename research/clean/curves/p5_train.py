"""Phase 5 reference: family F reduced to kept members K, inside the real network. Trainable: the kept members' 3b
kernels + biases (rows of their branch conv) and their 4a input columns + 4a input biases. Everything else frozen
(gradient masks). Start: 'derived' (original kernels + Bayes write) or 'random' (random kernels of the same norms,
bias set to the member's own firing rate, random columns of the same norms). Adam lr = 0.03 x each tensor's entry
scale (same relative rate as family.py). Early stop on dev.
usage: p5_train.py <family csv> <keep csv> <derived|random> [steps] [tag]"""
import sys, json, time, numpy as np, torch
from inception import *
from family import Data, IN4A, kl
from p5_drop import fam_stats, patched, evaluate_patched

BR = [("mixed3b_1x1_pre_relu_conv", 0, 128), ("mixed3b_3x3_pre_relu_conv", 128, 320),
      ("mixed3b_5x5_pre_relu_conv", 320, 416), ("mixed3b_pool_reduce_pre_relu_conv", 416, 480)]
F_ = [int(t) for t in sys.argv[1].split(",")]; keep = [int(t) for t in sys.argv[2].split(",")]; start = sys.argv[3]
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 1500; tag = sys.argv[5] if len(sys.argv) > 5 else f"p5-{start}-{'-'.join(map(str, keep))}"
torch.manual_seed(0); rng = np.random.default_rng(0)
m = load_model(); D = Data(seed=0, ndev=1000); st = fam_stats(m, F_, D)
for n, (w, b) in zip(IN4A, patched(m, F_, keep, st, "bayes")): getattr(m, n).weight.data.copy_(w); getattr(m, n).bias.data.copy_(b)
groups, masks = [], []
def add(t, mask):
    t.requires_grad_(True); mk = mask.to(t.dtype); masks.append(t.register_hook(lambda g, mk=mk: g * mk))
    scale = float((t.detach().abs() * mk).sum() / mk.sum().clamp(min=1)) + 1e-8; groups.append(dict(params=[t], lr=0.03 * scale))
for name, lo, hi in BR:
    rows = [c - lo for c in keep if lo <= c < hi]
    if not rows: continue
    conv = getattr(m, name); mw = torch.zeros_like(conv.weight); mw[rows] = 1; mb = torch.zeros_like(conv.bias); mb[rows] = 1
    if start == "random":
        with torch.no_grad():
            for r in rows:
                w0 = conv.weight[r]; conv.weight[r] = torch.randn_like(w0) * w0.norm() / np.sqrt(w0.numel())
    add(conv.weight, mw); add(conv.bias, mb)
for name in IN4A:
    conv = getattr(m, name); mw = torch.zeros_like(conv.weight); mw[:, keep] = 1
    if start == "random":
        with torch.no_grad():
            for c in keep:
                w0 = conv.weight[:, c]; conv.weight[:, c] = torch.randn_like(w0) * w0.norm() / np.sqrt(w0.numel())
    add(conv.weight, mw); add(conv.bias, torch.ones_like(conv.bias))
if start == "random":   # bias of each kept member set so it fires at its own intact rate (keeps random neurons alive)
    with torch.no_grad():
        ins = inputs_3b(m, to_3a(m, prep(D.X[np.sort(D.tr[:64])])))
        z = pre_3b(m, ins[0])
        for name, lo, hi in BR:
            for c in keep:
                if lo <= c < hi:
                    rate = float(st[0][1][F_.index(c), F_.index(c)] > 0) * 0.12
                    getattr(m, name).bias[c - lo] -= torch.quantile(z[:, c].reshape(-1)[::7], 1 - rate)
opt = torch.optim.Adam(groups); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
def dev_kl():
    with torch.no_grad():
        ks = []
        for i in range(0, len(D.dv), 100):
            b = D.dv[i:i + 100]; lg = from_3b(m, F.relu(pre_3b(m, to_3a(m, prep(D.X[b])))))
            ks.append(kl(torch.from_numpy(D.T[b].astype(np.float32)).to(DEV), lg).cpu().numpy())
        k = np.concatenate(ks); return float(k.mean()), float(k.std() / np.sqrt(len(k)))
print("start:", dev_kl(), flush=True)
best = (1e9, None, 0); bad = 0; run = []; t0 = time.time(); hist = []
trainable = [g["params"][0] for g in groups]
for step in range(1, steps + 1):
    b = np.sort(rng.choice(D.tr, 64, replace=False)); flip = rng.random(64) < 0.5
    xb = np.asarray(D.X[b]); xb = np.where(flip[:, None, None, None], xb[:, :, ::-1], xb)
    with torch.no_grad():
        a3 = to_3a(m, prep(xb)); t = torch.from_numpy(np.where(flip[:, None], D.TF[b], D.T[b]).astype(np.float32)).to(DEV)
    loss = kl(t, from_3b(m, F.relu(pre_3b(m, a3)))).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step(); run.append(loss.item())
    if step % 250 == 0 or step == steps:
        d, se = dev_kl(); hist.append(dict(step=step, train_kl=float(np.mean(run)), dev_kl=d, dev_se=se, sec=round(time.time() - t0))); run = []
        print(json.dumps(hist[-1]), flush=True)
        if d < best[0]: best = (d, [p.detach().clone() for p in trainable], step); bad = 0
        else:
            bad += 1
            if bad >= 4: break
with torch.no_grad():
    for p, v in zip(trainable, best[1]): p.copy_(v)
torch.save(dict(names=[n for n, _, _ in BR] + IN4A, state={n: (getattr(m, n).weight.detach().cpu(), getattr(m, n).bias.detach().cpu()) for n in [b[0] for b in BR] + IN4A}, F=F_, keep=keep),
           f"cache/ckpt/{tag}.pt")
json.dump(dict(best=best[0], step=best[2], hist=hist), open(f"../../../results/curves-{tag}.json", "w"), indent=1)
print(f"BEST {tag} {best[0]:.4f} at step {best[2]}", flush=True)
