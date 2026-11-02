"""D1: the strict baseline, zero images in the build (DERIVED). Images only score it (dev-5000, paired)."""
import numpy as np, torch
from inception import *
from family import Data, Family, forward_family, kl
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"])
IN4A = ["mixed4a_1x1_pre_relu_conv", "mixed4a_3x3_bottleneck_pre_relu_conv",
        "mixed4a_5x5_bottleneck_pre_relu_conv", "mixed4a_pool_reduce_pre_relu_conv"]
W5 = m.mixed3b_5x5_pre_relu_conv.weight.detach()[[i - 320 for i in fam]]; b5 = m.mixed3b_5x5_pre_relu_conv.bias.detach()[[i - 320 for i in fam]]
colsum = [getattr(m, n).weight.detach()[:, fam, 0, 0].sum(1, keepdim=True) for n in IN4A]
def moved_kernel(s):
    ks = 5 + 2 * int(np.abs(s).max()); c = ks // 2; K = torch.zeros(32, ks, ks, device=DEV)
    for i, (dy, dx) in enumerate(s): K[:, c + dy - 2:c + dy + 3, c + dx - 2:c + dx + 3] += W5[i]
    return K, ks
s = np.load("cache/c1_shifts.npz")["s"]; s = np.round(s - np.round(s.mean(0))).astype(int)
builds = {}
for tag, sh in [("moved", s), ("unmoved", np.zeros_like(s))]:
    K, ks = moved_kernel(sh); fm = Family(m, fam, 1, ks=ks).to(DEV)
    fm.set_weights(K[None], b5.sum()[None], colsum); builds[tag] = fm
    bal = (K.square().sum() + b5.sum() ** 2) / sum(c.square().sum() for c in colsum)
    print(f"{tag}: balance in^2/out^2 = {float(bal):.3f} (members: 1.0-1.17)")
fm = Family(m, fam, 1).to(DEV); fm.set_weights(torch.zeros(1, 32, 5, 5, device=DEV), torch.tensor([-1.0], device=DEV), [c * 0 for c in colsum]); builds["removed"] = fm
with torch.no_grad():
    ins = inputs_3b(m, to_3a(m, prep(D.X[np.sort(D.tr[:100])])))
    for k, f in builds.items(): print(k, "on-rate %.4f" % float((f.pre(ins) > 0).float().mean()), flush=True)
Kl = {k: [] for k in builds}
with torch.no_grad():
    for i in range(0, len(D.dv), 100):
        b = D.dv[i:i + 100]; ins = inputs_3b(m, to_3a(m, prep(D.X[b]))); t = torch.from_numpy(D.T[b].astype(np.float32)).to(DEV)
        for k, f in builds.items(): Kl[k].append(kl(t, forward_family(m, f, ins)).cpu().numpy())
Kl = {k: np.concatenate(v) for k, v in Kl.items()}
for k, v in Kl.items(): print(f"{k:8s} dev-{len(v)} KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})")
for a, b in [("moved", "removed"), ("moved", "unmoved")]:
    d = Kl[a] - Kl[b]; print(f"paired {a} - {b}: {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})")
np.savez("cache/d1_strict.npz", **Kl)
