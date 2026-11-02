"""C4 (check only, 100 train images). Is the trained k=1 neuron's pre-activation, after removing its mean, the same
signal as the weights-only moved sum (C1 shifts, and r=1 PREF shifts)? Prediction (written first): centred
pre-activation corr >= .8 with the C1 moved sum; the trained baseline (mean pre) is positive while the moved sum's is negative."""
import numpy as np, torch
from inception import *
from family import Data, Family
from derive import PREF
m = load_model(); D = Data(); fam = list(np.load("cache/s1_fine.npz")["fam"])
W5 = m.mixed3b_5x5_pre_relu_conv.weight.detach().cpu()[[i - 320 for i in fam]]
def moved_kernel(shifts, ks=11):
    K = torch.zeros(32, ks, ks); c = ks // 2
    for i, (dy, dx) in enumerate(shifts):
        dy, dx = int(round(dy)), int(round(dx)); K[:, c + dy - 2:c + dy + 3, c + dx - 2:c + dx + 3] += W5[i]
    return K
s1 = np.load("cache/c1_shifts.npz")["s"]; s1 = np.round(s1 - np.round(s1.mean(0)))
kr1 = moved_kernel([(np.sin(t), -np.cos(t)) for t in PREF]); kc1 = moved_kernel(s1); k0 = moved_kernel([(0, 0)] * 10)
idx = np.sort(np.random.default_rng(5).choice(D.tr, 100, replace=False))
tr = {}
for tag in ["L-k1-r-lr3e-2", "L-k1-m379-lr3e-2"]:
    fm = Family(m, fam, 1).to(DEV); st = torch.load(f"cache/ckpt/{tag}.pt", map_location="cpu"); fm.load_state_dict(st["state"] if "state" in st else st); tr[tag] = fm
zs = {k: [] for k in ["r", "m379", "c1", "r1", "zero", "famsum"]}
with torch.no_grad():
    for i in range(0, 100, 20):
        ins = inputs_3b(m, to_3a(m, prep(D.X[idx[i:i + 20]])))
        zs["r"].append(tr["L-k1-r-lr3e-2"].pre(ins)[:, 0]); zs["m379"].append(tr["L-k1-m379-lr3e-2"].pre(ins)[:, 0])
        for k, K in [("c1", kc1), ("r1", kr1), ("zero", k0)]:
            zs[k].append(F.conv2d(F.pad(ins[2], (5, 5, 5, 5)), K[None].to(DEV))[:, 0])
        zs["famsum"].append(F.relu(pre_3b(m, ins[0])[:, fam]).sum(1))
Z = {k: torch.cat(v)[:, 4:24, 4:24].reshape(-1).cpu().double() for k, v in zs.items()}
def cc(a, b): a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (a.norm() * b.norm()))
for t in ["r", "m379"]:
    print(f"trained {t}: mean pre {float(Z[t].mean()):+.0f} sd {float(Z[t].std()):.0f} | centred corr with moved sum C1 {cc(Z[t], Z['c1']):.3f}, r=1 {cc(Z[t], Z['r1']):.3f}, unmoved sum {cc(Z[t], Z['zero']):.3f}, post-ReLU family sum {cc(Z[t], Z['famsum']):.3f}")
for k in ["c1", "r1", "zero"]:
    print(f"moved sum {k}: mean pre {float(Z[k].mean()):+.0f} sd {float(Z[k].std()):.0f}, on-rate at bias 0 {float((Z[k] > 0).double().mean()):.3f}")
print("trained r vs trained m379 centred corr %.3f" % cc(Z["r"], Z["m379"]))
