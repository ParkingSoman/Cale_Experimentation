"""Mean-removed version of oc_kept2 (photo-specific message only). What training chooses to lose, measured in the stream after block 0 (LOG, 2026-11-23, tool fix after oc_kept.py
failed its power check). A's and B's block-0 outputs share their basis (same downsample, same head).
For each A stage-4 block-0 neuron i (all 512), on its own top NTOP photos (held + test, by pre-ReLU peak):
  c_i = neuron i's write in A: its output map through bn2-scaled conv2 (no shift), 512 x 7 x 7
  e   = B's block-0 write total minus A's (pre-ReLU, the shared downsample cancels)
  dropped_i = sum <e, -c_i> / sum <c_i, c_i> (plain dot product over channels and cells); kept_i = 1 - dropped_i
  shuffled control: e of the next photo in the set. Also the family (top 11 children of s3:69) as one summed write.
Predictors as in oc_kept.py (saved there). No class labels are read.
Saved: results/coarse/r18/oc_kept3.npz.  Usage: python -u oc_kept3.py [n_neurons]
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
P, NTOP, NMEAN = 69, 200, 2000
BTAGS = ("B128aug_s0", "B128aug_s1")


def bn_parts(bn):
    sc = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return sc, bn.bias - bn.running_mean * sc


def main():
    smoke = int(sys.argv[1]) if len(sys.argv) > 1 else None
    A = net_for("A"); b0 = A.layer4[0]
    Bs = {t: net_for(t).layer4[0] for t in BTAGS}
    s1, _ = bn_parts(b0.bn1)
    s2, _ = bn_parts(b0.bn2)
    R = b0.conv1.weight * s1.view(-1, 1, 1, 1)
    W2 = b0.conv2.weight * s2.view(-1, 1, 1, 1)                 # 512 out x 512 in x 3 x 3, bn2 scale folded
    wnorm = torch.sqrt((W2 ** 2).sum((0, 2, 3)))
    score = (R[:, P].sum((1, 2)) * wnorm).numpy()
    kids = np.argsort(-score)[:11]

    pos0 = np.load(os.path.join(r18.OUT, "read_A.npz"))["pos0"].astype(np.float32)
    caches = [r18.Cache("held"), r18.Cache("test")]
    nh = len(caches[0])

    def get(rows):
        xs = []
        for c, rr in ((caches[0], rows[rows < nh]), (caches[1], rows[rows >= nh] - nh)):
            if len(rr):
                xs.append(c.get(rr)[0])
        return torch.cat(xs)

    def write0(bl, x):
        return bl.bn2(bl.conv2(torch.relu(bl.bn1(bl.conv1(x)))))   # block-0 write total, pre-ReLU, no downsample

    def one_write(inner, idx):
        return F.conv2d(inner[:, idx], W2[:, idx], padding=1)      # write of the neurons idx (no bn2 shift)

    mrows = np.sort(np.random.default_rng(3).choice(len(pos0), NMEAN, replace=False))
    mean_c = torch.zeros(512, 512, 7, 7)                      # per neuron: mean of its write map (channel x cell)
    mean_e = {t: torch.zeros(512, 7, 7) for t in BTAGS}
    for s in range(0, NMEAN, 250):
        x = get(mrows[s:s + 250])
        inner = torch.relu(b0.bn1(b0.conv1(x)))
        m_in = inner.sum(0) / NMEAN                            # 512 x 7 x 7, mean output map of each neuron
        mean_c += torch.stack([F.conv2d(m_in[j:j + 1].unsqueeze(0), W2[:, j:j + 1], padding=1)[0] for j in range(512)])
        wa = b0.bn2(b0.conv2(inner))
        for t, bl in Bs.items():
            mean_e[t] += (write0(bl, x) - wa).sum(0) / NMEAN
    print("means done", flush=True)
    N = smoke or 512
    out = {}
    for t in BTAGS:
        for k in ("dropped", "shuf", "cosE"):
            out[f"{k}_{t}"] = np.zeros(N)
    out["cnorm"] = np.zeros(N)
    t0 = time.time()
    for i in range(N + 1):                                         # the last pass is the family of 11 children
        fam = i == N
        if fam:
            rows = np.sort(np.argsort(-np.maximum.reduce([pos0[:, k] for k in kids]))[:NTOP])
            idx = torch.as_tensor(kids.copy())
        else:
            rows = np.sort(np.argsort(-pos0[:, i])[:NTOP])
            idx = torch.as_tensor([i])
        x = get(rows)
        inner = torch.relu(b0.bn1(b0.conv1(x)))
        c = one_write(inner, idx) - mean_c[idx].sum(0)
        wa = b0.bn2(b0.conv2(inner))
        cc = float((c * c).sum())
        res = {}
        for t, bl in Bs.items():
            e = write0(bl, x) - wa - mean_e[t]
            res[t] = (float((e * -c).sum()) / cc, float((torch.roll(e, 1, 0) * -c).sum()) / cc,
                      float((e * -c).sum() / (e.norm() * c.norm())))
        if fam:
            out.update({f"family_{k}_{t}": v for t in BTAGS for k, v in zip(("dropped", "shuf", "cosE"), res[t])})
            print("family (11 children, summed write):", {t: [round(v, 3) for v in res[t]] for t in BTAGS})
            continue
        out["cnorm"][i] = cc / len(rows)
        for t in BTAGS:
            out[f"dropped_{t}"][i], out[f"shuf_{t}"][i], out[f"cosE_{t}"][i] = res[t]
        if i % 64 == 0:
            print(f"  neuron {i}: dropped s0 {res[BTAGS[0]][0]:.3f} s1 {res[BTAGS[1]][0]:.3f} "
                  f"shuf {res[BTAGS[0]][1]:.3f} {time.time() - t0:.0f} s", flush=True)

    def sp(a, b):
        return np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]

    pk = np.load(os.path.join(r18.OUT, "oc_kept.npz"))
    iqr = lambda v: np.percentile(v, 75) - np.percentile(v, 25)
    for t in BTAGS:
        dr, sh = out[f"dropped_{t}"][:N], out[f"shuf_{t}"][:N]
        kept = 1 - dr
        print(f"[{t}] dropped median {np.median(dr):.3f} IQR {iqr(dr):.3f} | shuffled median {np.median(sh):.3f} "
              f"IQR {iqr(sh):.3f}")
        print(f"   kept deciles: {np.percentile(kept, np.arange(10, 100, 10)).round(2).tolist()}")
        for name, v in (("worth", pk["worth"]), ("frequency", pk["freq"]), ("write norm", pk["wnorm"]),
                        ("co-firing", pk["cofire"]), ("parent link", pk["plink"]), ("R1 score", pk["score"])):
            print(f"   Spearman(kept, {name}) {sp(kept, v[:N]):+.3f}")
        if N == 512:
            print(f"   11 children: Spearman(kept, parent link) {sp(kept[kids], pk['plink'][kids]):+.3f}")
            print("   children (neuron, kept):", [(int(k), round(float(kept[k]), 2)) for k in kids])
    print(f"Spearman(kept s0, kept s1) {sp(out['dropped_B128aug_s0'][:N], out['dropped_B128aug_s1'][:N]):+.3f} | "
          f"shuffled s0 vs s1 {sp(out['shuf_B128aug_s0'][:N], out['shuf_B128aug_s1'][:N]):+.3f}")
    fn = os.path.join(r18.OUT, "oc_kept3" + ("_smoke" if smoke else "") + ".npz")
    np.savez(fn, kids=kids, **out)
    print("saved", fn)


if __name__ == "__main__":
    main()
