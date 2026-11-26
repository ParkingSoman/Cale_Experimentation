# scale: neurons=s3:69, A:children; net=A; photos=train-cache, held, test; where=cluster
# rung: Q=rung 2: train the coarse neuron's read AND write (same one-unit form); N=1 unit (the parent), 2 seeds; EST=25 min GPU; PREV=results/oc_morph2.npz; PERMIT=Shonu 2026-11-24 "the trained coarse neuron's read should be trained as well. we are testing both the coarse neuron's write and read rule"
"""Rung 2: the trained coarse neuron, read and write (LOG "Rung 2: train the read too").

The coarse unit is the parent, stage-3 stream channel 69 (after the last stage-3 block's add and ReLU):
  p = ReLU( conv(t, w69) + b69 + s69 )
  t   = stage-3 block-1 hidden (256 x 14 x 14, after its ReLU), fixed
  s69 = the skip into channel 69 (stage-3 block-0 output, channel 69), fixed
  w69 = block-1 conv2 row 69 with bn2 scale folded (256 x 3 x 3), b69 = bn2 shift of channel 69
The READ = (w69, b69). Trained here; the calculated rule keeps it unchanged. Every reader of channel 69 (the other 501
stage-4 block-0 neurons, the downsample) sees the new p (rule: other readers keep reading the parent).
The WRITE = the same form as oc_morph.py: 11 kernels on p + 11 write columns into block 0's residual; the 11 R1
children are deleted. Start: read = A's own, write kernels random, write columns 0 (as the write-only run).
Trained by KL(A || changed A) on train + trainflip; lr from {1e-4, 3e-4, 1e-3} by held KL (300 steps); 600 steps,
cosine decay, checkpoint by held KL every 25 steps. Report on the oc_morph test rows.
Step "cache" (once): saves t and s69 per photo for train, trainflip, held and the test rows (fp16).
Usage: python -u oc_morph_rw.py [smoke]
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import r18
from capture18 import net_for

P, K_N, SEEDS = 69, 11, (0, 1)
SMOKE = "smoke" in sys.argv
LR_W = 3e-4                                      # write lr: the write-only run's chosen lr (oc_morph2)
LRS_R = (0.0, 1e-3, 3e-3, 1e-2, 3e-2)            # read lr sweep (run 1 used one shared lr: held KL rose; see LOG)
SWEEP, STEPS, BS, EVERY = (20, 40, 64, 20) if SMOKE else (300, 600, 256, 25)
DEV = r18.DEV
TC = os.path.join(r18.CACHE, "s3b1")


def bn_parts(bn):
    sc = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return sc, bn.bias - bn.running_mean * sc


def make_cache(a, test_rows):
    os.makedirs(TC, exist_ok=True)
    ph = r18.net.Photos("read")
    b = a.layer3[1]
    for name in ("train", "trainflip", "held", "test"):
        f_t = os.path.join(TC, f"{name}_t.npy")
        if os.path.exists(f_t):
            continue
        c = r18.Cache(name)
        rows = test_rows if name == "test" else np.arange(len(c))
        ix = c.ix[rows]
        T = np.lib.format.open_memmap(f_t + ".tmp.npy", "w+", np.float16, (len(ix), 256, 14, 14))
        S = np.zeros((len(ix), 14, 14), np.float16)
        worst = 0.0
        with torch.no_grad():
            for s in range(0, len(ix), 200):
                x = ph.tensor(ix[s:s + 200])
                if name == "trainflip":
                    x = torch.flip(x, dims=[3])
                h0 = a.layer3[0](a.layer2(a.layer1(r18.net.stem(a, x.to(DEV)))))
                t = torch.relu(b.bn1(b.conv1(h0)))
                out = torch.relu(b.bn2(b.conv2(t)) + h0)
                T[s:s + len(t)] = t.half().cpu().numpy()
                S[s:s + len(t)] = h0[:, P].half().cpu().numpy()
                xc = torch.as_tensor(np.ascontiguousarray(c.x[rows[s:s + 200]]), device=DEV).float()
                worst = max(worst, float((out - xc).abs().max()))
        T.flush(); del T
        os.replace(f_t + ".tmp.npy", f_t)
        np.save(os.path.join(TC, f"{name}_s69.npy"), S)
        np.save(os.path.join(TC, f"{name}_rows.npy"), rows)
        print(f"cache {name}: {len(ix)} photos; rebuilt stage-3 output vs cache max abs diff {worst:.4f}", flush=True)


class TCache:
    """Stage-3 output cache plus t and s69 for the same rows."""
    def __init__(self, name, gpu=False):
        self.c = r18.Cache(name, gpu=gpu)
        self.rows = np.load(os.path.join(TC, f"{name}_rows.npy"))
        t = np.load(os.path.join(TC, f"{name}_t.npy"), mmap_mode="r")
        self.t = torch.from_numpy(np.ascontiguousarray(t)).to(DEV) if gpu else t
        self.s = torch.from_numpy(np.load(os.path.join(TC, f"{name}_s69.npy"))).to(DEV)
        self.gpu = gpu

    def __len__(self):
        return len(self.rows)

    def get(self, i):                       # i = positions in self.rows
        i = np.sort(np.asarray(i))          # r18.Cache.get sorts its rows; t and s69 must follow the same order
        assert np.all(np.diff(self.rows) > 0)
        x, z = self.c.get(self.rows[i])
        t = self.t[torch.as_tensor(i, device=DEV)].float() if self.gpu else \
            torch.from_numpy(np.ascontiguousarray(self.t[i])).to(DEV).float()
        return x, z, t, self.s[torch.as_tensor(i, device=DEV)].float().unsqueeze(1)


def main():
    t0 = time.time()
    m = net_for("A").to(DEV).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    mo = np.load(os.path.join(r18.OUT, "oc_morph2.npz"))
    strong, rand = mo["test_strong_rows"], mo["test_random_rows"]
    test_rows = np.union1d(strong, rand)
    with torch.no_grad():
        make_cache(m, test_rows)
    b0, b1 = m.layer4[0], m.layer4[1]
    s1, _ = bn_parts(b0.bn1)
    s2, _ = bn_parts(b0.bn2)
    R = b0.conv1.weight * s1.view(-1, 1, 1, 1)
    V = b0.conv2.weight * s2.view(-1, 1, 1, 1)
    wnorm = torch.sqrt((V ** 2).sum((0, 2, 3)))
    order = np.argsort(-(R[:, P].sum((1, 2)) * wnorm).cpu().numpy())
    assert np.array_equal(order[:K_N], mo["order"][:K_N])
    K = torch.as_tensor(order[:K_N].copy(), device=DEV)
    k_cal, v_cal = R[K][:, P:P + 1].contiguous(), V[:, K].contiguous()
    b3 = m.layer3[1]
    s3s, s3b = bn_parts(b3.bn2)
    w69_0 = (b3.conv2.weight[P] * s3s[P]).unsqueeze(0).contiguous()               # 1 x 256 x 3 x 3
    b69_0 = s3b[P].view(1)

    def run(x, t, s69, mode, w69=None, b69=None, kk=None, vv=None):
        if w69 is not None:                                                        # new read -> new channel 69
            p_new = torch.relu(F.conv2d(t, w69, padding=1) + b69.view(1, 1, 1, 1) + s69)
            x = torch.cat([x[:, :P], p_new, x[:, P + 1:]], 1)
        inner = torch.relu(b0.bn1(b0.conv1(x)))
        if mode != "base":
            inner = inner.clone(); inner[:, K] = 0
        r = b0.bn2(b0.conv2(inner))
        if mode == "morph":
            r = r + F.conv2d(F.conv2d(x[:, P:P + 1], kk, stride=2, padding=1), vv, padding=1)
        h1 = torch.relu(r + b0.downsample(x))
        return r18.net.head(m, r18.net.block_parts(b1, h1)["h"])

    gpu = DEV == "cuda"
    tr, trf = TCache("train", gpu=gpu), TCache("trainflip", gpu=gpu)
    held, test = TCache("held"), TCache("test")
    pos = {int(r): i for i, r in enumerate(test.rows)}
    sets = {"test_strong": np.array([pos[int(r)] for r in strong]), "test_random": np.array([pos[int(r)] for r in rand])}
    held_i = np.arange(len(held))[::(31 if SMOKE else 1)]
    if SMOKE:
        sets = {k: v[:40] for k, v in sets.items()}
    print(f"caches ready; {time.time()-t0:.0f} s", flush=True)

    with torch.no_grad():                                                          # sanity: read unchanged = A
        x, z, t, s = held.get(np.arange(50))
        e_read = float(r18.kl_rows(z, run(x, t, s, "base", w69_0, b69_0)).max())
        e_cal = float((r18.kl_rows(z, run(x, t, s, "morph", w69_0, b69_0, k_cal, v_cal))
                       - r18.kl_rows(z, run(x, t, s, "morph", None, None, k_cal, v_cal))).abs().max())
    print(f"sanity: original read rebuilt, KL max {e_read:.2e} (bar 1e-3, fp16 cache); calc with rebuilt read vs cache "
          f"read, KL diff max {e_cal:.2e}", flush=True)
    assert e_read < 1e-3

    @torch.no_grad()
    def kl_on(c, idx, **kw):
        out = []
        for s in range(0, len(idx), 250):
            x, z, t, s69 = c.get(idx[s:s + 250])
            out.append(r18.kl_rows(z, run(x, t, s69, **kw)).cpu().numpy())
        return np.concatenate(out)

    @torch.enable_grad()
    def train(seed, lr_r, steps, lr=LR_W):
        g = torch.Generator(device="cpu").manual_seed(seed)
        kk = torch.empty(K_N, 1, 3, 3); torch.nn.init.kaiming_normal_(kk, mode="fan_in", nonlinearity="linear", generator=g)
        kk = kk.to(DEV).requires_grad_()
        vv = torch.zeros(512, K_N, 3, 3, device=DEV, requires_grad=True)
        w69 = w69_0.clone().requires_grad_()
        b69 = b69_0.clone().requires_grad_()
        opt = torch.optim.AdamW([{"params": [kk, vv], "lr": lr, "base": lr},
                                 {"params": [w69, b69], "lr": lr_r, "base": lr_r}], weight_decay=0.0)
        rng = np.random.default_rng(100 + seed)
        best, curve = (np.inf, None), []
        for step in range(1, steps + 1):
            for gr in opt.param_groups:
                gr["lr"] = gr["base"] * 0.5 * (1 + np.cos(np.pi * (step - 1) / steps))
            c = tr if rng.random() < 0.5 else trf
            x, z, t, s69 = c.get(rng.choice(len(c), BS, replace=False))
            y = run(x, t, s69, "morph", w69, b69, kk, vv)
            loss = F.kl_div(F.log_softmax(y, 1), F.log_softmax(z, 1), log_target=True, reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
            if step % EVERY == 0 or step == steps:
                prm = dict(w69=w69.detach(), b69=b69.detach(), kk=kk.detach(), vv=vv.detach())
                hk = float(kl_on(held, held_i, mode="morph", **prm).mean())
                curve.append((step, float(loss.detach()), hk))
                print(f"    seed {seed} lr_r {lr_r:g} step {step} loss {float(loss):.5f} held KL {hk:.5f} {time.time()-t0:.0f} s",
                      flush=True)
                if hk < best[0]:
                    best = (hk, {k: v.clone() for k, v in prm.items()})
        return best, np.array(curve)

    sweep = {lr_r: train(0, lr_r, STEPS)[0][0] for lr_r in LRS_R}
    lr = min(sweep, key=sweep.get)
    print(f"read-lr sweep (write lr {LR_W:g}, {STEPS} steps) {sweep} -> lr_r {lr:g}", flush=True)
    res = dict(order=order, lr_r=lr, lr_w=LR_W, sweep_lr_r=np.array(LRS_R), sweep_held=np.array([sweep[v] for v in LRS_R]),
               w69_0=w69_0.cpu().numpy(), b69_0=b69_0.cpu().numpy())
    trained = {}
    for seed in SEEDS:
        (h, prm), curve = train(seed, lr, STEPS)
        trained[seed] = prm
        res.update({f"rw{seed}_{k}": v.cpu().numpy() for k, v in prm.items()})
        res[f"rw{seed}_curve"], res[f"rw{seed}_held"] = curve, h
        dw = prm["w69"] - w69_0
        print(f"seed {seed}: read change |dw|/|w0| {float(dw.norm() / w69_0.norm()):.3f}, cos(w, w0) "
              f"{float((prm['w69'] * w69_0).sum() / prm['w69'].norm() / w69_0.norm()):.3f}, b69 {float(b69_0):.3f} -> "
              f"{float(prm['b69']):.3f}", flush=True)

    wo = {s: (torch.as_tensor(mo[f"tr{s}_k"], device=DEV), torch.as_tensor(mo[f"tr{s}_v"], device=DEV)) for s in SEEDS}
    for name, idx in sets.items():
        kl = {"delete": kl_on(test, idx, mode="delete"),
              "calc": kl_on(test, idx, mode="morph", kk=k_cal, vv=v_cal)}
        for s in SEEDS:
            kl[f"write_only{s}"] = kl_on(test, idx, mode="morph", kk=wo[s][0], vv=wo[s][1])
            kl[f"read_write{s}"] = kl_on(test, idx, mode="morph", **trained[s])
        n = len(idx)
        B = [np.random.default_rng(i).integers(0, n, n) for i in range(2000)]
        rec = lambda c, b: 1 - kl[c][b].mean() / kl["delete"][b].mean()
        line = [f"[{name}] n {n}"]
        for c in kl:
            if c == "delete":
                continue
            ci = np.percentile([rec(c, b) for b in B], [2.5, 97.5])
            line.append(f"{c} {rec(c, np.arange(n)):.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        for s in SEEDS:
            d = np.percentile([rec(f"read_write{s}", b) - rec(f"write_only{s}", b) for b in B], [2.5, 97.5])
            line.append(f"read_write{s}-write_only{s} CI [{d[0]:.3f}, {d[1]:.3f}]")
        print(" | ".join(line), flush=True)
        res.update({f"{name}_{c}": v for c, v in kl.items()})
        res[f"{name}_rows"] = idx
    out = os.path.join(r18.OUT, "oc_morph_rw5" + ("_smoke" if SMOKE else "") + ".npz")
    np.savez(out, **res)
    print("saved", out, f"{time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
