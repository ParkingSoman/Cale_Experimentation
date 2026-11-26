"""What training chooses to lose (LOG, 2026-11-23, section "What training chooses to lose").
For each A stage-4 block-0 neuron i (all 512), on its own top 100 photos (held + test, by pre-ReLU peak from read_A.npz):
  d = A's logits with i deleted - A's logits;  e = B's logits - A's logits (B = B128aug_s0 and B128aug_s1)
  dropped_i = sum <e, d>_F / sum <d, d>_F (Fisher at A's softmax); kept_i = 1 - dropped_i
  worth_i = mean KL(A || A-minus-i); secondary = (KL(A-i || B) - KL(A || B)) / symmetric KL(A, A-i)
  shuffled control: d of photo p paired with e of the next photo in the set.
Predictors: firing frequency, write norm, co-firing (mean of 10 largest peak correlations), parent link (corr with
s3:69's peak), R1 child score. No class labels are read.
Runs on the cluster in n6/clean/neuron/coarse. Saved: results/coarse/r18/oc_kept.npz.
Usage: python -u oc_kept.py [n_neurons]   (a number = smoke run on the first n neurons)
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import r18
from capture18 import net_for

torch.set_grad_enabled(False)
P, NTOP = 69, 100
BTAGS = ("B128aug_s0", "B128aug_s1")


def bn_parts(bn):
    sc = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return sc, bn.bias - bn.running_mean * sc


def fisher(p, u, v):
    return (p * u * v).sum(1) - (p * u).sum(1) * (p * v).sum(1)


def kl(za, zb):
    return r18.kl_rows(za, zb)


def main():
    smoke = int(sys.argv[1]) if len(sys.argv) > 1 else None
    A = net_for("A"); b0, b1 = A.layer4[0], A.layer4[1]
    Bs = {t: net_for(t) for t in BTAGS}
    s1, _ = bn_parts(b0.bn1)
    s2, _ = bn_parts(b0.bn2)
    R = b0.conv1.weight * s1.view(-1, 1, 1, 1)
    wnorm = torch.sqrt(((b0.conv2.weight * s2.view(-1, 1, 1, 1)) ** 2).sum((0, 2, 3)))
    score = (R[:, P].sum((1, 2)) * wnorm).numpy()

    rd = np.load(os.path.join(r18.OUT, "read_A.npz"))
    pos0 = rd["pos0"].astype(np.float32)                      # 19,000 x 512, held then test
    caches = [r18.Cache("held"), r18.Cache("test")]
    nh = len(caches[0])
    assert pos0.shape[0] == nh + len(caches[1])
    ppeak = np.concatenate([np.concatenate([np.asarray(c.x[s:s + 1000, P], dtype=np.float32).max((1, 2))
                                            for s in range(0, len(c), 1000)]) for c in caches])
    freq = (pos0 > 0).mean(0)
    Z = (pos0 - pos0.mean(0)) / (pos0.std(0) + 1e-6)
    C = Z.T @ Z / len(Z); np.fill_diagonal(C, -np.inf)
    cofire = np.sort(C, 1)[:, -10:].mean(1)
    pz = (ppeak - ppeak.mean()) / ppeak.std()
    plink = (Z * pz[:, None]).mean(0)
    print(f"photos {len(pos0)}, predictors done", flush=True)

    def get(rows):
        xs, zs = [], []
        for c, rr in ((caches[0], rows[rows < nh]), (caches[1], rows[rows >= nh] - nh)):
            if len(rr):
                x, z = c.get(rr); xs.append(x); zs.append(z)
        return torch.cat(xs), torch.cat(zs)

    def tail(x, inner):
        h1 = torch.relu(b0.bn2(b0.conv2(inner)) + b0.downsample(x))
        return r18.net.head(A, r18.net.block_parts(b1, h1)["h"])

    N = smoke or 512
    out = {k: np.zeros(N) for k in ("worth", "base")}
    for t in BTAGS:
        for k in ("dropped", "dropped_shuf", "sec", "errB", "pdropped", "pdropped_shuf"):
            out[f"{k}_{t}"] = np.zeros(N)
    t0 = time.time()
    for i in range(N):
        rows = np.sort(np.argsort(-pos0[:, i])[:NTOP])
        x, z = get(rows)
        inner = torch.relu(b0.bn1(b0.conv1(x)))
        ya = tail(x, inner)
        inn = inner.clone(); inn[:, i] = 0
        yd = tail(x, inn)
        p = F.softmax(z, 1)
        d = yd - z
        out["base"][i] = float(kl(z, ya).max())
        out["worth"][i] = float(kl(z, yd).mean())
        sym = 0.5 * float((kl(z, yd) + kl(yd, z)).mean())
        dd = float(fisher(p, d, d).sum())
        zc = z - z.mean(1, keepdim=True)                          # A's own logits: the confidence direction

        def strip(u):
            return u - (fisher(p, u, zc) / fisher(p, zc, zc)).unsqueeze(1) * zc
        dp = strip(d); ddp = float(fisher(p, dp, dp).sum())
        for t, m in Bs.items():
            yb = r18.run4(m, x)
            e = yb - z
            out[f"dropped_{t}"][i] = float(fisher(p, e, d).sum()) / dd
            out[f"dropped_shuf_{t}"][i] = float(fisher(p, torch.roll(e, 1, 0), d).sum()) / dd
            ep = strip(e)
            out[f"pdropped_{t}"][i] = float(fisher(p, ep, dp).sum()) / ddp
            out[f"pdropped_shuf_{t}"][i] = float(fisher(p, torch.roll(ep, 1, 0), dp).sum()) / ddp
            out[f"sec_{t}"][i] = float((kl(yd, yb) - kl(z, yb)).mean()) / sym
            out[f"errB_{t}"][i] = float(kl(z, yb).mean())
        if i % 32 == 0:
            print(f"  neuron {i}: worth {out['worth'][i]:.4f} dropped s0 {out['dropped_B128aug_s0'][i]:.2f} "
                  f"s1 {out['dropped_B128aug_s1'][i]:.2f} shuf {out['dropped_shuf_B128aug_s0'][i]:.2f} "
                  f"base {out['base'][i]:.1e} {time.time() - t0:.0f} s", flush=True)

    def spearmanr(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return (np.corrcoef(ra, rb)[0, 1],)

    sl = slice(0, N)
    print(f"base KL max {out['base'].max():.2e}")
    kids = np.argsort(-score)[:11]
    for t, (dk, sk) in [(t, kk) for t in BTAGS for kk in (("dropped", "dropped_shuf"), ("pdropped", "pdropped_shuf"))]:
        print(f"== measure {dk}")
        dr, sh = out[f"{dk}_{t}"], out[f"{sk}_{t}"]
        kept = 1 - dr
        iqr = lambda v: np.percentile(v, 75) - np.percentile(v, 25)
        print(f"[{t}] dropped median {np.median(dr):.3f} IQR {iqr(dr):.3f} | shuffled median {np.median(sh):.3f} "
              f"IQR {iqr(sh):.3f} | B error KL on these photos median {np.median(out[f'errB_{t}']):.3f}")
        for name, v in (("worth", out["worth"]), ("frequency", freq[sl]), ("write norm", wnorm.numpy()[sl]),
                        ("co-firing", cofire[sl]), ("parent link", plink[sl]), ("R1 score", score[sl])):
            print(f"   Spearman(kept, {name}) {spearmanr(kept, v)[0]:+.3f}")
        print(f"   Spearman(kept, secondary) {spearmanr(kept, out[f'sec_{t}'])[0]:+.3f}")
        if N == 512:
            print(f"   11 children: Spearman(kept, parent link) {spearmanr(kept[kids], plink[kids])[0]:+.3f}")
            print("   children (neuron, kept, worth, parent link):",
                  [(int(k), round(float(kept[k]), 2), round(float(out['worth'][k]), 4), round(float(plink[k]), 2))
                   for k in kids])
    for dk in ("dropped", "pdropped"):
        print(f"{dk}: Spearman(kept s0, kept s1) {spearmanr(out[f'{dk}_B128aug_s0'], out[f'{dk}_B128aug_s1'])[0]:+.3f}")
    fn = os.path.join(r18.OUT, "oc_kept" + ("_smoke" if smoke else "") + ".npz")
    np.savez(fn, score=score, wnorm=wnorm.numpy(), freq=freq, cofire=cofire, plink=plink, **out)
    print("saved", fn)


if __name__ == "__main__":
    main()
