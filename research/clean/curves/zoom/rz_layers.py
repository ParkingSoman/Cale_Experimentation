# serves: H-map
"""Which ResNet-18 layer looks most like a map of the A.7 claim types? (zoom plan, step 1, after rz_mapcheck.py)

For 3000 train-split images (3 per class) and five layers, per channel (pooled over positions, as a reader that pools
would see it; for inner layers this is a description only, their readers are convolutions):
  frac_zero_units   share of (image, channel, position) entries that are exactly 0
  frac_zero_pooled  share of (image, channel) pooled values that are exactly 0
  eff_classes       median effective number of classes of the pooled activity
  top10 / top50     median share of activity in the channel's top 10 / 50 classes
  nn_corr           median nearest-neighbour correlation of the channel's values over images x positions
Layers: layer3 output (256 x 14 x 14), layer4.0 inner (after its first relu), layer4.0 output, layer4.1 inner, layer4.1
output (the last hidden layer).
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import resnets as R

m = R.load("resnet18")
X, y = R.load_images()
tr, _, _ = R.split(y)
rank = np.zeros(len(y), int)
for c in range(1000):
    ix = np.where(y == c)[0]; rank[ix] = np.arange(len(ix))
ix = np.sort(tr[rank[tr] < 3]); L = y[ix]
acc = {k: [] for k in ("layer3", "l40_inner", "l40_out", "l41_inner", "l41_out")}
with torch.no_grad():
    for b in range(0, len(ix), 100):
        x = R.prep(X[ix[b:b + 100]])
        h = m.layer3(m.layer2(m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(x)))))))
        acc["layer3"].append(h.cpu().numpy().astype(np.float16))
        b0 = m.layer4[0]
        i0 = b0.relu(b0.bn1(b0.conv1(h))); acc["l40_inner"].append(i0.cpu().numpy().astype(np.float16))
        o0 = b0(h); acc["l40_out"].append(o0.cpu().numpy().astype(np.float16))
        b1 = m.layer4[1]
        i1 = b1.relu(b1.bn1(b1.conv1(o0))); acc["l41_inner"].append(i1.cpu().numpy().astype(np.float16))
        acc["l41_out"].append(b1(o0).cpu().numpy().astype(np.float16))
res = {}
for k, v in acc.items():
    a = np.concatenate(v).astype(np.float32); n, C = a.shape[:2]; a = a.reshape(n, C, -1)
    p = a.mean(2)
    A = np.stack([p[L == c].mean(0) for c in range(1000)], 1)
    S = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
    ent = np.exp(-(S * np.log(S + 1e-12)).sum(1)); ss = np.sort(S, 1)[:, ::-1]
    sub = a[:, :, :: max(1, a.shape[2] // 49)].transpose(1, 0, 2).reshape(C, -1)
    sub = sub - sub.mean(1, keepdims=True); sub /= np.sqrt((sub ** 2).mean(1, keepdims=True)) + 1e-12
    G = sub @ sub.T / sub.shape[1]; np.fill_diagonal(G, -9)
    res[k] = {"channels": C, "frac_zero_units": float((a == 0).mean()), "frac_zero_pooled": float((p == 0).mean()),
              "eff_classes_median": float(np.median(ent)), "top10_median": float(np.median(ss[:, :10].sum(1))),
              "top50_median": float(np.median(ss[:, :50].sum(1))), "nn_corr_median": float(np.median(G.max(1)))}
    print(k, {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in res[k].items()}, flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_rz_layers.json"), "w"), indent=1)
