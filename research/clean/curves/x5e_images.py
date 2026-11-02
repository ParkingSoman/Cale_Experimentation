"""R5' part 4: images with the largest level-0 cross term; per-position contribution maps m(p) = Cov_c(s_c(p), s_c) for
level 0 and the build trained from level 0 (sum over p = 2 x L3), and where the curve family fires."""
import numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from inception import *
import b3_train
from family import Data, kl, pool3, IN4A
exec(open("x2b_reason.py").read().split("names = {")[0])          # m, D, idx, convs, cont, reader_pre
fam = list(np.load("cache/s1_fine.npz")["fam"]); K = 10
r = np.load("cache/x2c_blocks.npz"); cross = r["level0_full"] - r["level0_spatial"] - r["level0_position"] + r["level0_diag"]
crosst = r["trained_from_level0_full"] - r["trained_from_level0_spatial"] - r["trained_from_level0_position"] + r["trained_from_level0_diag"]
order = np.argsort(-cross); top = order[:12]
print("cross term share carried by the top 5% of images:", round(float(np.sort(cross)[::-1][:25].sum() / cross.sum()), 3))
print("top-12 dev images (index, class, level0 cross, trained cross):")
names = {}
try:
    import json; names = {i: n for i, n in enumerate(json.load(open("imagenet_classes.json")))}
except Exception: pass
def load(v):
    fm = b3_train.FamilyRA(m, fam, 1); fm.add_rb(m); fm.add_ra(m)
    if v.startswith("x2_"): fm.load_state_dict(torch.load(f"cache/{v}.pt", map_location="cpu")); return fm.to(DEV)
    return b3_train.load_b3(m, v)
B = {"level0": load("x2_a_level0_x"), "trained": load("B3-k0-formulastart")}
b = idx[top]
with torch.no_grad():
    x = pool3(F.relu(pre_3b(m, inputs_3b(m, to_3a(m, prep(D.X[b])))[0])))
    p = F.max_pool2d(F.pad(x, (1, 1, 1, 1), value=float('-inf')), 3, 1); a0 = reader_pre(x, p)
    E = {k: [u - v for u, v in zip(a0, reader_pre(x, p, fm))] for k, fm in B.items()}
    hF = x[:, fam].sum(1).cpu().numpy()                                         # curve family activity (14 x 14)
pre = [t.clone().requires_grad_(True) for t in a0]; lg = cont([F.relu(t) for t in pre])
tp = lg.detach().softmax(1).topk(K, 1); pk = tp.values / tp.values.sum(1, keepdim=True)
J = []
for c in range(K):
    J.append(torch.autograd.grad(lg.gather(1, tp.indices[:, c:c + 1]).sum(), pre, retain_graph=c < K - 1))
maps = {}
with torch.no_grad():
    for k in B:
        sp = torch.stack([sum((J[c][j] * E[k][j]).sum(1) for j in range(4)) for c in range(K)], 1)   # (B, K, 14, 14)
        s = sp.sum((2, 3)); spc = sp - (pk[:, :, None, None] * sp).sum(1, keepdim=True); sc = s - (pk * s).sum(1, keepdim=True)
        maps[k] = (pk[:, :, None, None] * spc * sc[:, :, None, None]).sum(1).cpu().numpy()          # (B, 14, 14), sums to 2 L3
lab = D.lab
for n, i in enumerate(top):
    cls = int(D.y[idx[i]]); print(f"  {n:2d} dev#{idx[i]} class {cls} {names.get(cls, '')}  cross {cross[i]:.3f}  trained {crosst[i]:.3f}  KL0 {r['level0_full'][i]:.3f}")
fig, ax = plt.subplots(4, 12, figsize=(30, 10.5))
for n in range(12):
    img = np.asarray(D.X[b[n]]); ax[0, n].imshow(img); ax[0, n].set_title(f"#{n} cls {int(D.y[b[n]])}", fontsize=9)
    ax[1, n].imshow(img); ax[1, n].imshow(hF[n], cmap="viridis", alpha=.6, extent=[0, 224, 224, 0]); ax[1, n].set_title("curve family fires", fontsize=8)
    v = np.abs(maps["level0"][n]).max()
    for row, k in [(2, "level0"), (3, "trained")]:
        ax[row, n].imshow(img); ax[row, n].imshow(maps[k][n], cmap="bwr", vmin=-v, vmax=v, alpha=.65, extent=[0, 224, 224, 0])
        ax[row, n].set_title(f"{k}: sum {maps[k][n].sum():.2f}", fontsize=8)
for a in ax.flat: a.axis("off")
plt.tight_layout(); plt.savefig("cache/x5_images.png", dpi=55); print("saved cache/x5_images.png")
