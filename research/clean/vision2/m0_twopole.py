"""M0: two-pole gate check. Rules: VISION-LOG.md M0."""
import json
import numpy as np
from sklearn.cluster import KMeans
from convnext import *
from acts import get_acts

X, y = load_images(); names = [n.split(",")[0] for n in class_names()]
amax, amin = get_acts(); ana, _ = split(y); ya = y[ana]
F = np.load(os.path.join(HF, "imagenet", "feat_tiny_ana20k.npy")); F = F / np.linalg.norm(F, axis=1, keepdims=True)
K = 1000; g = KMeans(K, n_init=1, random_state=0).fit_predict(F); cnt = np.bincount(g, minlength=K)[:, None]
P, N, MX, MN = {}, {}, {}, {}
for b in amax:
    mx_ = np.zeros((K, amax[b].shape[1]), np.float32); np.add.at(mx_, g, amax[b]); mx_ /= cnt
    mn_ = np.zeros((K, amin[b].shape[1]), np.float32); np.add.at(mn_, g, amin[b]); mn_ /= cnt
    m1 = np.median(mx_, 0); P[b] = (mx_ - m1) / (mx_.max(0) - m1 + 1e-9) >= .5
    m2 = np.median(mn_, 0); N[b] = (m2 - mn_) / (m2 - mn_.min(0) + 1e-9) >= .5
    MX[b], MN[b] = mx_, mn_
R = json.load(open(os.path.join(RES, "vision-r10c-signed.json"))); out = {}
for bs, d in R.items():
    b = int(bs); both, pos = [], []
    for r in d["rows"]:
        n, p = r["child"], r["parent"]; neg = N[b + 1][:, n]
        cov = (P[b][:, p] & neg).sum() / max(neg.sum(), 1)
        (both if cov >= .6 else pos if cov < .3 else []).append(r["ig"] > 0)
    out[b] = {"both_n": len(both), "both_pos": float(np.mean(both)) if both else None, "posonly_n": len(pos), "posonly_pos": float(np.mean(pos)) if pos else None}
    print(f"block {b}->{b+1}: BOTH-POLE n={len(both)} IG positive {out[b]['both_pos']} | POS-ONLY n={len(pos)} IG positive {out[b]['posonly_pos']}")
# search a both-pole parent for 17 n2283
n = 2283; pp, nn_ = P[17][:, n], N[17][:, n]
cls = lambda mask: sorted({names[c] for c in np.bincount(ya[np.isin(g, np.where(mask)[0])], minlength=1000).argsort()[::-1][:8]})
print("\nn2283 positive groups:", pp.sum(), "classes", cls(pp)); print("n2283 negative groups:", nn_.sum(), "classes", cls(nn_))
for b in (14, 15, 16):
    cp = (P[b] & pp[:, None]).sum(0) / pp.sum(); cn = (P[b] & nn_[:, None]).sum(0) / nn_.sum(); sc = np.minimum(cp, cn); br = P[b].sum(0)
    for k in np.argsort(-sc)[:3]:
        print(f"  block {b} n{k}: covers + {cp[k]:.2f}, covers - {cn[k]:.2f}, breadth {br[k]} groups; top classes {cls(P[b][:, k])}")
    out[f"search_{b}"] = [{"neuron": int(k), "cov_pos": float(cp[k]), "cov_neg": float(cn[k]), "breadth": int(br[k])} for k in np.argsort(-sc)[:3]]
json.dump(out, open(os.path.join(RES, "vision-m0-twopole.json"), "w"), indent=1)
