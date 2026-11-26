# scale: neurons=s3:69, A:children; net=A; photos=train-cache, held, test; where=cluster
# rung: Q=the parent becomes the coarse neuron, calculated vs trained write; N=1 unit (the parent), 2 seeds; EST=20 min GPU; PERMIT=Shonu approved plan "The parent becomes the coarse neuron" 2026-11-23 (rung 1 only)
"""The parent becomes the coarse neuron (plan 2026-11-23, LOG section of the same name).

Object (one unit): in A, the 11 R1 children of s3:69 are deleted. The parent s3:69 is the coarse neuron. Its activation p
is its own value (channel 69 of the stage-3 output), unchanged: no new kernel, bias or ReLU. Its other readers (the
other 501 block-0 neurons, the downsample) read channel 69 as before. The parent gets one new write W(p) into block 0's
residual output r (before the ReLU), linear in p.
  Calculated: W(p) = sum_i v_i * (k_i * p). k_i = child i's channel-69 kernel (bn1 scale folded, 3x3, stride 2),
              v_i = child i's write column (bn2 scale folded, 512 x 3x3). Each child keeps only the part the parent
              drives: its other inputs, its bias and its ReLU are gone. Nothing fitted. No constant shift.
  Trained:    the same form (11 linear kernels on channel 69, 11 write columns), kernels random, write columns 0 at the
              start (= delete). Trained by KL(A || changed A). The parent's read stays frozen.
Comparison row only: the old two-unit build (OCN: a new unit with its own kernel, bias, ReLU between parent and writes).
Data: A's stage-3 caches. Train on train + trainflip, choose lr and checkpoint on held, report on test.
Runs on the cluster in n6/clean/neuron/coarse. Saved: results/coarse/r18/oc_morph.npz.
Usage: python -u oc_morph.py [smoke] [tune2]
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
TUNE2 = "tune2" in sys.argv          # rerun 2026-11-23: lower lr range, shorter schedule, finer checkpoints
LRS = (1e-4, 3e-4, 1e-3) if TUNE2 else (1e-3, 3e-3, 1e-2)
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
SWEEP, STEPS, BS, EVERY = (20, 40, 64, 20) if SMOKE else ((300, 600, 256, 25) if TUNE2 else (300, 2500, 256, 250))
DEV = r18.DEV


def bn_parts(bn):
    sc = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return sc, bn.bias - bn.running_mean * sc


def main():
    t0 = time.time()
    m = net_for("A").to(DEV).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    b0, b1 = m.layer4[0], m.layer4[1]
    s1, t1 = bn_parts(b0.bn1)
    s2, _ = bn_parts(b0.bn2)
    R = b0.conv1.weight * s1.view(-1, 1, 1, 1)                                   # 512 x 256 x 3 x 3, bn1 scale in
    V = b0.conv2.weight * s2.view(-1, 1, 1, 1)                                   # 512 x 512 x 3 x 3, bn2 scale in
    wnorm = torch.sqrt((V ** 2).sum((0, 2, 3)))
    score = (R[:, P].sum((1, 2)) * wnorm).cpu().numpy()
    order = np.argsort(-score)
    K = torch.as_tensor(order[:K_N].copy(), device=DEV)
    print("children:", order[:K_N].tolist(), flush=True)

    # calculated morph: the children's channel-69 kernels and write columns
    k_cal = R[K][:, P:P + 1].contiguous()                                        # 11 x 1 x 3 x 3
    v_cal = V[:, K].contiguous()                                                 # 512 x 11 x 3 x 3

    # old two-unit build (OCN of oc_tcn / oc_nuudle3), comparison row only
    a = torch.as_tensor(score[order[:K_N]], dtype=torch.float32, device=DEV)
    w = a / a.sum()
    ocn_k = (k_cal * w.view(-1, 1, 1, 1)).sum(0, keepdim=True)
    ocn_b = (t1[K] * w).sum().view(1)
    ocn_v = v_cal.sum(1, keepdim=True)

    def run(x, mode, kk=None, vv=None):
        inner = torch.relu(b0.bn1(b0.conv1(x)))
        if mode == "base":
            r = b0.bn2(b0.conv2(inner))
        else:
            inner = inner.clone(); inner[:, K] = 0
            r = b0.bn2(b0.conv2(inner))                                          # the other 501 neurons + bn2 shift
            p = x[:, P:P + 1]
            if mode == "morph":                                                  # the parent writes directly
                r = r + F.conv2d(F.conv2d(p, kk, stride=2, padding=1), vv, padding=1)
            elif mode == "ocn":
                r = r + F.conv2d(torch.relu(F.conv2d(p, ocn_k, stride=2, padding=1) + ocn_b), ocn_v, padding=1)
            elif mode == "children_back":                                        # wiring check: exact children
                g = torch.relu(F.conv2d(x, R[K], stride=2, padding=1) + t1[K].view(1, -1, 1, 1))
                r = r + F.conv2d(g, v_cal, padding=1)
        h1 = torch.relu(r + b0.downsample(x))
        return r18.net.head(m, r18.net.block_parts(b1, h1)["h"])

    gpu = DEV == "cuda"
    tr, trf = r18.Cache("train", gpu=gpu), r18.Cache("trainflip", gpu=gpu)
    held, test = r18.Cache("held"), r18.Cache("test")
    print(f"caches: train {len(tr)}, trainflip {len(trf)}, held {len(held)}, test {len(test)}; dev {DEV}; "
          f"{time.time()-t0:.0f} s", flush=True)

    def peaks(c):
        out = []
        for s in range(0, len(c), 1000):
            xs = c.x[s:s + 1000, P]
            xs = xs if torch.is_tensor(xs) else torch.as_tensor(np.ascontiguousarray(xs))
            out.append(xs.float().amax((1, 2)).cpu().numpy())
        return np.concatenate(out)

    held_rows = np.arange(len(held))[::(31 if SMOKE else 1)]
    te_pk, tr_pk = peaks(test), peaks(tr)
    sets = {"test_strong": (test, np.nonzero(te_pk >= 2.92)[0]),
            "test_random": (test, np.sort(np.random.default_rng(2).choice(len(test), 2000, replace=False))),
            "train_strong": (tr, np.nonzero(tr_pk >= 2.92)[0])}
    if SMOKE:
        sets = {k: (c, r[:40]) for k, (c, r) in sets.items()}

    @torch.no_grad()
    def kl_on(cache, rows, mode, **kw):
        out = []
        for s in range(0, len(rows), 250):
            x, z = cache.get(rows[s:s + 250])
            out.append(r18.kl_rows(z, run(x, mode, **kw)).cpu().numpy())
        return np.concatenate(out)

    # sanity: base reproduces the cache; putting the children back through the new wiring reproduces A;
    # the calculated kernels on channel 69 alone equal channel 69's share of the children's read
    with torch.no_grad():
        x, z = test.get(sets["test_strong"][1][:100])
        base_max = float(r18.kl_rows(z, run(x, "base")).max())
        back_max = float(r18.kl_rows(z, run(x, "children_back")).max())
        x69 = torch.zeros_like(x); x69[:, P] = x[:, P]
        share_err = float((F.conv2d(x69, R[K], stride=2, padding=1)
                           - F.conv2d(x[:, P:P + 1], k_cal, stride=2, padding=1)).abs().max())
    print(f"sanity: base KL max {base_max:.2e}, children back KL max {back_max:.2e} (bars 1e-4); "
          f"ch69 share error {share_err:.2e} (bar 1e-3)", flush=True)
    assert base_max < 1e-4 and back_max < 1e-4 and share_err < 1e-3

    @torch.enable_grad()
    def train(seed, lr, steps):
        g = torch.Generator(device="cpu").manual_seed(seed)
        kk = torch.empty(K_N, 1, 3, 3); torch.nn.init.kaiming_normal_(kk, mode="fan_in", nonlinearity="linear", generator=g)
        kk = kk.to(DEV).requires_grad_()
        vv = torch.zeros(512, K_N, 3, 3, device=DEV, requires_grad=True)          # start = delete
        opt = torch.optim.AdamW([kk, vv], lr=lr, weight_decay=0.0)
        rng = np.random.default_rng(100 + seed)
        best, curve = (np.inf, None), []
        for step in range(1, steps + 1):
            for gr in opt.param_groups:
                gr["lr"] = lr * 0.5 * (1 + np.cos(np.pi * (step - 1) / steps))
            c = tr if rng.random() < 0.5 else trf
            x, z = c.get(rng.choice(len(c), BS, replace=False))
            y = run(x, "morph", kk, vv)
            loss = F.kl_div(F.log_softmax(y, 1), F.log_softmax(z, 1), log_target=True, reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
            if step % EVERY == 0 or step == steps:
                hk = float(kl_on(held, held_rows, "morph", kk=kk.detach(), vv=vv.detach()).mean())
                curve.append((step, float(loss), hk))
                print(f"    seed {seed} lr {lr:g} step {step} loss {float(loss):.5f} held KL {hk:.5f} "
                      f"{time.time()-t0:.0f} s", flush=True)
                if hk < best[0]:
                    best = (hk, (kk.detach().clone(), vv.detach().clone()))
        return best, np.array(curve)

    hk = {"delete": float(kl_on(held, held_rows, "delete").mean()),
          "morph_cal": float(kl_on(held, held_rows, "morph", kk=k_cal, vv=v_cal).mean()),
          "ocn": float(kl_on(held, held_rows, "ocn").mean())}
    print("held KL:", {k: round(v, 5) for k, v in hk.items()}, flush=True)
    sweep = {}
    for lr in LRS:
        (h, _), _ = train(0, lr, SWEEP)
        sweep[lr] = h
    lr = min(sweep, key=sweep.get)
    print(f"sweep {sweep} -> lr {lr:g}", flush=True)

    # the parent's write as one map: response of the stage-4 stream to a unit impulse of p at one stage-3 cell
    def impulse(kk, vv):
        d = torch.zeros(1, 1, 14, 14, device=DEV); d[0, 0, 6, 6] = 1.0
        with torch.no_grad():
            return F.conv2d(F.conv2d(d, kk, stride=2, padding=1), vv, padding=1)[0].cpu().numpy()   # 512 x 7 x 7

    res = dict(order=order, score=score, lr=lr, sweep_lrs=np.array(LRS), sweep_kl=np.array([sweep[v] for v in LRS]),
               **{f"held_{k}": v for k, v in hk.items()},
               k_cal=k_cal.cpu().numpy(), v_cal=v_cal.cpu().numpy(), imp_cal=impulse(k_cal, v_cal))
    trained = {}
    for seed in SEEDS:
        (h, prm), curve = train(seed, lr, STEPS)
        trained[seed] = prm
        res.update({f"tr{seed}_k": prm[0].cpu().numpy(), f"tr{seed}_v": prm[1].cpu().numpy(),
                    f"tr{seed}_curve": curve, f"tr{seed}_held": h, f"tr{seed}_imp": impulse(*prm)})

    def cos(u, v):
        u, v = u.ravel(), v.ravel()
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
    for seed in SEEDS:
        print(f"write map cosine: calculated vs trained seed {seed} {cos(res['imp_cal'], res[f'tr{seed}_imp']):.3f}; "
              f"norm ratio trained/calc {np.linalg.norm(res[f'tr{seed}_imp']) / np.linalg.norm(res['imp_cal']):.3f}")
    print(f"write map cosine: seed 0 vs seed 1 {cos(res['tr0_imp'], res['tr1_imp']):.3f}", flush=True)

    for name, (cache, rows) in sets.items():
        kl = {"base": kl_on(cache, rows, "base"), "delete": kl_on(cache, rows, "delete"),
              "morph_cal": kl_on(cache, rows, "morph", kk=k_cal, vv=v_cal), "ocn": kl_on(cache, rows, "ocn")}
        for seed, (kk, vv) in trained.items():
            kl[f"morph_tr{seed}"] = kl_on(cache, rows, "morph", kk=kk, vv=vv)
        n = len(rows)
        B = [np.random.default_rng(i).integers(0, n, n) for i in range(2000)]

        def rec(c, b):
            return 1 - kl[c][b].mean() / kl["delete"][b].mean()
        line = [f"[{name}] photos {n}, base max {kl['base'].max():.1e}, delete {kl['delete'].mean():.5f}"]
        for c in ["morph_cal", "ocn"] + [f"morph_tr{s}" for s in SEEDS]:
            ci = np.percentile([rec(c, b) for b in B], [2.5, 97.5])
            line.append(f"{c} {rec(c, np.arange(n)):.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        for c in ["ocn"] + [f"morph_tr{s}" for s in SEEDS]:
            d = np.percentile([rec(c, b) - rec("morph_cal", b) for b in B], [2.5, 97.5])
            line.append(f"{c}-morph_cal CI [{d[0]:.3f}, {d[1]:.3f}]")
        print(" | ".join(line), flush=True)
        res.update({f"{name}_rows": rows, **{f"{name}_{c}": v for c, v in kl.items()}})

    out = os.path.join(r18.OUT, "oc_morph" + ("2" if TUNE2 else "") + ("_smoke" if SMOKE else "") + ".npz")
    np.savez(out, **res)
    print("saved", out, f"{time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
