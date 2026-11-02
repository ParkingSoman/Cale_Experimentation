# serves: H-dialfit
"""Part C measurement for Hypothesis H-dialfit (COARSE-THEORY.md A.3): is the trained 3b coarse curve unit a dial whose
pre-activation, on its own region, is the least-squares affine fit of the family's output seen through its own write
(the target t of Corollary T38.2)?

Model: InceptionV1 (lucent), checkpoint cache/ckpt/fam-curves-k1-lr0.03-s0-random.pt (the k = 1 crush of the ten
curve units, family.py). No training. 300 train images; all 28 x 28 positions for the firing fraction; 100 random
positions per image for the fit.
Per position: x = the 800 numbers of the 5 x 5 window of the 32-channel 5 x 5 bottleneck (what the coarse unit and the
members read); z_c = the coarse pre-activation; h = the ten members' outputs relu(z_i).
Targets: "sum" = sum_i h_i (first version, T38.2 with an all-ones write); "write" = t = c^T C h / |c|^2, with C the
members' original columns into the four 4a input convolutions and c the coarse unit's trained column (T38.2).
Measures, per target:
  (1) firing fraction of the coarse unit over all positions (dial if most);
  (2) on the coarse unit's firing positions, split by image (fit / held out): least-squares affine fit target ~ x;
      held-out R^2 of that fit;
  (3) held-out correlation of z_c with the fit's prediction, and the slope of z_c on it (T38.2 predicts 1);
  (4) cosine of the coarse kernel with the fit's kernel, against 1000 random Gaussian kernels (null).
Prediction for "sum" (written before the first run): (1) above 0.9; (3) at least 0.8; (4) above the null's 99th
percentile. Near-true if (1) and (3) hold. Result of the first run: (1) 0.83, (3) 0.37, (4) 0.094 against 0.100: not
near-true.
Prediction for "write" (written before its run, after the first run): (3) correlation at least 0.8 and slope in
[1/2, 2]; (4) above the null's 99th percentile.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from family import Data, Family
from inception import CURVES, DEV, inputs_3b, load_model, prep, to_3a
from family import IN4A

HERE = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0)
rng = np.random.default_rng(0)

m = load_model()
for p in m.parameters():
    p.requires_grad_(False)
D = Data(seed=0)
ck = torch.load(os.path.join(HERE, "cache", "ckpt", "fam-curves-k1-lr0.03-s0-random.pt"), map_location="cpu",
                weights_only=False)
fm = Family(m, CURVES, 1, init="random", ks=5)
fm.load_state_dict(ck["state"])
fm = fm.to(DEV).eval()
conv5 = m.mixed3b_5x5_pre_relu_conv
fam5 = [i - 320 for i in CURVES]

# write target weights: t = w^T h with w = C^T c / |c|^2
C = torch.cat([getattr(m, n).weight.detach()[:, CURVES, 0, 0] for n in IN4A]).cpu().double()   # (R, 10)
cw = torch.cat(fm.weights()[2]).cpu().double()                                                # (R, 1)
w_write = (C.T @ cw / (cw.T @ cw)).flatten().numpy()
print("write target weights on the ten members:", np.round(w_write, 4).tolist())
print(f"cosine of the coarse column with the members' columns: "
      f"{np.round((C.T @ cw).flatten().numpy() / C.norm(dim=0).numpy() / float(cw.norm()), 3).tolist()}")

idx = np.sort(D.tr[:300])
X_list, zc_list, h_list, img_list = [], [], [], []
n_fire, n_all = 0, 0
with torch.no_grad():
    for k in range(0, len(idx), 20):
        b = idx[k:k + 20]
        ins = inputs_3b(m, to_3a(m, prep(D.X[b])))
        c = ins[2]
        zc = fm.pre(ins)[:, 0]                                   # (B, 28, 28)
        h = F.relu(conv5(F.pad(c, (2, 2, 2, 2)))[:, fam5])       # (B, 10, 28, 28)
        n_fire += int((zc > 0).sum()); n_all += zc.numel()
        P = F.unfold(F.pad(c, (2, 2, 2, 2)), 5)                  # (B, 800, 784)
        for j in range(len(b)):
            pos = rng.choice(784, 100, replace=False)
            X_list.append(P[j][:, pos].T.cpu().numpy())
            zc_list.append(zc[j].flatten()[pos].cpu().numpy())
            h_list.append(h[j].flatten(1)[:, pos].T.cpu().numpy())
            img_list.append(np.full(100, k + j))
X = np.concatenate(X_list).astype(np.float64); zc = np.concatenate(zc_list).astype(np.float64)
H = np.concatenate(h_list).astype(np.float64); img = np.concatenate(img_list)
fire_frac = n_fire / n_all
print(f"(1) firing fraction of the coarse unit over all positions: {fire_frac:.4f}")

act = zc > 0
fit_img = img % 2 == 0
tr, te = act & fit_img, act & ~fit_img
A = np.hstack([X, np.ones((len(X), 1))])
G = A[tr].T @ A[tr]
lam = 1e-6 * np.trace(G) / A.shape[1]
wk = (fm.read.detach() * fm.wscale).flatten().cpu().numpy().astype(np.float64)
out = dict(fire_frac=fire_frac, n_positions=int(len(X)), n_active_test=int(te.sum()), w_write=w_write.tolist())
for name, y in (("sum", H.sum(1)), ("write", H @ w_write)):
    beta = np.linalg.solve(G + lam * np.eye(A.shape[1]), A[tr].T @ y[tr])
    pred = A @ beta
    r2 = 1 - np.sum((y[te] - pred[te]) ** 2) / np.sum((y[te] - y[te].mean()) ** 2)
    corr = float(np.corrcoef(zc[te], pred[te])[0, 1])
    slope = float(np.polyfit(pred[te], zc[te], 1)[0])
    bk = beta[:800]
    cos = float(wk @ bk / np.linalg.norm(wk) / np.linalg.norm(bk))
    null = [abs(g @ bk) / np.linalg.norm(g) / np.linalg.norm(bk) for g in rng.normal(size=(1000, 800))]
    p99 = float(np.percentile(null, 99))
    print(f"[{name}] (2) held-out R^2 {r2:.4f}; (3) corr(z_c, fit) {corr:.4f}, slope {slope:.4f}; "
          f"(4) cos(coarse kernel, fit kernel) {cos:.4f}, null 99th pct {p99:.4f}")
    out[name] = dict(r2=float(r2), corr=corr, slope=slope, cos=cos, null_p99=p99)
ok_w = out["write"]["corr"] >= 0.8 and 0.5 <= out["write"]["slope"] <= 2 and out["write"]["cos"] > out["write"]["null_p99"]
out["write_prediction_holds"] = bool(ok_w)
print("prediction for the write target holds:", ok_w)
json.dump(out, open(os.path.join(HERE, "..", "..", "..", "results", "curves-pc-dialfit.json"), "w"), indent=1)
