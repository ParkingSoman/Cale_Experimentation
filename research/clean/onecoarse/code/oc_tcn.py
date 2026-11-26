# scale: neurons=s3:69, A:children, A:tcn; net=A; photos=train-cache, held, test; where=cluster
# rung: Q=trained coarse neuron with the same cut and readers as OCN; N=1 new neuron, 2 seeds; EST=20 min GPU; PERMIT=Shonu approved plan "same cut, same readers" 2026-11-23
"""SUPERSEDED 2026-11-23, never run: both builds here are TWO-UNIT objects (a new unit between the parent and the write;
TCN reads all 256 channels). Not Shonu's object. The same-cut pair is now oc_morph.py.

TCN against OCN: the same cut, the same readers; only the method differs (plan 2026-11-23, "same cut, same readers").

Cut: in A, the 11 R1 children of s3:69 (top 11 by child score) are replaced by ONE neuron. Everything else in A is frozen
(the other 501 block-0 neurons, block 1, the downsample, the head).
  OCN (calculated): read = channel 69 only, 3 x 3 shape = a-weighted mean of the children's bn1-scaled channel-69 kernels;
                    bias = a-weighted mean of their bn1 shifts; write = sum of their bn2-scaled conv2 columns.
  TCN (trained):    the same form (read 256 x 3 x 3, bias, write 512 x 3 x 3), random start, trained by KL(A || changed A).
Data: A's stage-3 caches. Train on train + trainflip (as B was), choose lr and checkpoint on held, report on test.
Runs on the cluster in n6/clean/neuron/coarse. Saved: results/coarse/r18/oc_tcn.npz.
Usage: python -u oc_tcn.py [smoke]
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import r18
from capture18 import net_for

P, K_N, SEEDS, LRS = 69, 11, (0, 1), (1e-3, 3e-3, 1e-2)
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
SWEEP, STEPS, BS, EVERY = (20, 40, 64, 20) if SMOKE else (300, 3000, 256, 250)
DEV = r18.DEV


def bn_parts(bn):
    sc = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return sc, bn.bias - bn.running_mean * sc


def main():
    t0 = time.time()
    m = net_for("A").to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    b0, b1 = m.layer4[0], m.layer4[1]
    s1, t1 = bn_parts(b0.bn1)
    s2, _ = bn_parts(b0.bn2)
    R = b0.conv1.weight * s1.view(-1, 1, 1, 1)
    wnorm = torch.sqrt(((b0.conv2.weight * s2.view(-1, 1, 1, 1)) ** 2).sum((0, 2, 3)))
    score = (R[:, P].sum((1, 2)) * wnorm).cpu().numpy()
    order = np.argsort(-score)
    K = torch.as_tensor(order[:K_N].copy(), device=DEV)
    a = torch.as_tensor(score[order[:K_N]], dtype=torch.float32, device=DEV)
    w = a / a.sum()
    print("children:", order[:K_N].tolist(), flush=True)

    # OCN in the shared form
    rf = (R[K] * w.view(-1, 1, 1, 1)).sum(0, keepdim=True)
    ocn_r = torch.zeros_like(rf); ocn_r[:, P] = rf[:, P]
    ocn_b = (t1[K] * w).sum().view(1)
    ocn_w = (b0.conv2.weight[:, K] * s2.view(-1, 1, 1, 1)).sum(1, keepdim=True)        # 512 x 1 x 3 x 3

    def run(x, rd=None, bias=None, wr=None, mode="neuron"):
        inner = torch.relu(b0.bn1(b0.conv1(x)))
        if mode == "base":
            r = b0.bn2(b0.conv2(inner))
        else:
            inner = inner.clone(); inner[:, K] = 0
            r = b0.bn2(b0.conv2(inner))
            if mode == "neuron":
                g = torch.relu(F.conv2d(x, rd, stride=2, padding=1) + bias.view(1, -1, 1, 1))
                r = r + F.conv2d(g, wr, padding=1)
        h1 = torch.relu(r + b0.downsample(x))
        return r18.net.head(m, r18.net.block_parts(b1, h1)["h"])

    gpu = DEV == "cuda"
    tr, trf = r18.Cache("train", gpu=gpu), r18.Cache("trainflip", gpu=gpu)
    held, test = r18.Cache("held"), r18.Cache("test")          # read from disk; only the training caches go to the GPU
    print(f"caches: train {len(tr)}, trainflip {len(trf)}, held {len(held)}, test {len(test)}; {time.time()-t0:.0f} s",
          flush=True)

    def peaks(c):
        return np.concatenate([torch.as_tensor(np.ascontiguousarray(c.x[s:s + 1000, P]) if not c.gpu else c.x[s:s + 1000, P])
                               .float().amax((1, 2)).cpu().numpy() for s in range(0, len(c), 1000)])

    held_rows = np.arange(len(held))[::(31 if SMOKE else 1)]
    te_pk, tr_pk = peaks(test), peaks(tr)
    sets = {"test_strong": (test, np.nonzero(te_pk >= 2.92)[0]),
            "test_random": (test, np.sort(np.random.default_rng(2).choice(len(test), 2000, replace=False))),
            "train_strong": (tr, np.nonzero(tr_pk >= 2.92)[0])}
    if SMOKE:
        sets = {k: (c, r[:40]) for k, (c, r) in sets.items()}

    @torch.no_grad()
    def kl_on(cache, rows, **kw):
        out = []
        for s in range(0, len(rows), 250):
            x, z = cache.get(rows[s:s + 250])
            out.append(r18.kl_rows(z, run(x, **kw)).cpu().numpy())
        return np.concatenate(out)

    # sanity: base reproduces cached logits; OCN in this form = OCN of oc_nuudle3 (children's outputs set to the fold)
    with torch.no_grad():
        x, z = test.get(sets["test_strong"][1][:100])
        base_max = float(r18.kl_rows(z, run(x, mode="base")).max())
        inner = torch.relu(b0.bn1(b0.conv1(x)))
        fo = torch.relu(F.conv2d(x, ocn_r, stride=2, padding=1) + ocn_b)
        inn = inner.clone(); inn[:, K] = fo
        y_ref = r18.net.head(m, r18.net.block_parts(b1, torch.relu(b0.bn2(b0.conv2(inn)) + b0.downsample(x)))["h"])
        y_new = run(x, ocn_r, ocn_b, ocn_w)
        form_err = float((y_ref - y_new).abs().max())
    print(f"sanity: base KL max {base_max:.2e} (bar 1e-4); OCN form error {form_err:.2e} (bar 1e-3)", flush=True)
    assert base_max < 1e-4 and form_err < 1e-3

    def train(seed, lr, steps):
        g = torch.Generator(device="cpu").manual_seed(seed)
        rd = torch.empty(1, 256, 3, 3); torch.nn.init.kaiming_normal_(rd, mode="fan_in", nonlinearity="relu", generator=g)
        wr = torch.empty(512, 1, 3, 3); torch.nn.init.kaiming_normal_(wr, mode="fan_out", nonlinearity="relu", generator=g)
        rd, wr = rd.to(DEV).requires_grad_(), wr.to(DEV).requires_grad_()
        bias = torch.zeros(1, device=DEV, requires_grad=True)
        opt = torch.optim.AdamW([rd, bias, wr], lr=lr, weight_decay=0.0)
        rng = np.random.default_rng(100 + seed)
        best, curve = (np.inf, None), []
        for step in range(1, steps + 1):
            for gr in opt.param_groups:
                gr["lr"] = lr * 0.5 * (1 + np.cos(np.pi * (step - 1) / steps))
            c = tr if rng.random() < 0.5 else trf
            x, z = c.get(rng.choice(len(c), BS, replace=False))
            y = run(x, rd, bias, wr)
            loss = F.kl_div(F.log_softmax(y, 1), F.log_softmax(z, 1), log_target=True, reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
            if step % EVERY == 0 or step == steps:
                hk = float(kl_on(held, held_rows, rd=rd.detach(), bias=bias.detach(), wr=wr.detach()).mean())
                curve.append((step, float(loss), hk))
                print(f"    seed {seed} lr {lr:g} step {step} loss {float(loss):.5f} held KL {hk:.5f} "
                      f"{time.time()-t0:.0f} s", flush=True)
                if hk < best[0]:
                    best = (hk, (rd.detach().clone(), bias.detach().clone(), wr.detach().clone()))
        return best, np.array(curve)

    hk_ocn = float(kl_on(held, held_rows, rd=ocn_r, bias=ocn_b, wr=ocn_w).mean())
    hk_del = float(kl_on(held, held_rows, mode="delete").mean())
    print(f"held KL: delete {hk_del:.5f}, OCN {hk_ocn:.5f}", flush=True)
    sweep = {}
    for lr in LRS:
        (hk, _), _ = train(0, lr, SWEEP)
        sweep[lr] = hk
    lr = min(sweep, key=sweep.get)
    print(f"sweep {sweep} -> lr {lr:g}", flush=True)

    res = dict(order=order, score=score, lr=lr, sweep_lrs=np.array(LRS), sweep_kl=np.array([sweep[v] for v in LRS]),
               held_delete=hk_del, held_ocn=hk_ocn,
               ocn_read=ocn_r.cpu().numpy(), ocn_bias=ocn_b.cpu().numpy(), ocn_write=ocn_w.cpu().numpy())
    tcn = {}
    for seed in SEEDS:
        (hk, params), curve = train(seed, lr, STEPS)
        tcn[seed] = params
        res.update({f"tcn{seed}_read": params[0].cpu().numpy(), f"tcn{seed}_bias": params[1].cpu().numpy(),
                    f"tcn{seed}_write": params[2].cpu().numpy(), f"tcn{seed}_curve": curve, f"tcn{seed}_held": hk})

    for name, (cache, rows) in sets.items():
        kl = {"base": kl_on(cache, rows, mode="base"), "delete": kl_on(cache, rows, mode="delete"),
              "ocn": kl_on(cache, rows, rd=ocn_r, bias=ocn_b, wr=ocn_w)}
        for seed, (rd, bias, wr) in tcn.items():
            kl[f"tcn{seed}"] = kl_on(cache, rows, rd=rd, bias=bias, wr=wr)
        n = len(rows)
        B = [np.random.default_rng(i).integers(0, n, n) for i in range(2000)]

        def rec(c, b):
            return 1 - kl[c][b].mean() / kl["delete"][b].mean()
        line = [f"[{name}] photos {n}, base max {kl['base'].max():.1e}, delete {kl['delete'].mean():.5f}"]
        for c in ["ocn"] + [f"tcn{s}" for s in SEEDS]:
            ci = np.percentile([rec(c, b) for b in B], [2.5, 97.5])
            line.append(f"{c} {rec(c, np.arange(n)):.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        for s in SEEDS:
            d = np.percentile([rec(f"tcn{s}", b) - rec("ocn", b) for b in B], [2.5, 97.5])
            line.append(f"tcn{s}-ocn CI [{d[0]:.3f}, {d[1]:.3f}]")
        print(" | ".join(line), flush=True)
        res.update({f"{name}_rows": rows, **{f"{name}_{c}": v for c, v in kl.items()}})

    out = os.path.join(r18.OUT, "oc_tcn" + ("_smoke" if SMOKE else "") + ".npz")
    np.savez(out, **res)
    print("saved", out, f"{time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
