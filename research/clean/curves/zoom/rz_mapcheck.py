# serves: H-map
"""Does ResNet-18's last hidden layer admit a map of the A.7 claim types at the channel level? (zoom plan, step 1)

Three measurements on the train split of the cache (cache/r18), all on what the logits read (pooled relu(z)):
 1. districts: per channel, the effective number of classes of its activity (exp of the entropy of its class shares),
    the share in its top 10 / 50 / 100 / 250 classes, and the share of its variance that class means explain
    (between-class variance / total). And the 'constant outside' form: per channel, with region = its top t classes,
    the variance outside the region / total variance (the DM3 form on h; 1 = no structure).
 2. families (proportional channels, DM6 item 5 / c(S)): nearest-neighbour cosine and correlation of z over
    images x positions, 4000 images; PCA dimensions of z for 50/80/90/99 % of the variance.
 3. strong parts (the essay: strong activations define the job): per channel keep the entries above its q-quantile
    (q = 0.9, 0.99, 0.999); class spread of the strong part, its share of activity, and the KL of replacing the weak part
    by its mean over P (bias re-fit), 12000 images.
Writes results_rz_mapcheck.json next to this file.
"""
import json, os
import numpy as np

C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "r18")
H = np.load(os.path.join(C, "h_train.npy")); lab = np.load(os.path.join(C, "lab_train.npy"))
W = np.load(os.path.join(C, "head.npz"))["W"]
Z = np.load(os.path.join(C, "z_train.npy"), mmap_mode="r")
out = {}
q5 = [.1, .25, .5, .75, .9]


def classtable(h, L):
    A = np.zeros((h.shape[1], 1000))
    for k in range(1000):
        if (L == k).any():
            A[:, k] = h[L == k].mean(0)
    S = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
    ent = np.exp(-(S * np.log(S + 1e-12)).sum(1))
    ss = np.sort(S, 1)[:, ::-1]
    return A, ent, ss


A, ent, ss = classtable(H, lab)
d = {"eff_classes_q": np.quantile(ent, q5).round(1).tolist(),
     "top_share_median": {t: float(np.median(ss[:, :t].sum(1))) for t in (10, 50, 100, 250)},
     "between_class_var_share_q": np.quantile(A.var(1) / H.var(0), q5).round(3).tolist(),
     "frac_pooled_zero": float((H == 0).mean())}
cons = {}
for t in (5, 20, 50, 100, 200):
    e = []
    for c in range(512):
        inR = np.isin(lab, np.argsort(A[c])[::-1][:t]); pout = (~inR).mean()
        e.append(pout ** 2 * H[~inR, c].var() / H[:, c].var())
    cons[t] = np.quantile(e, [.1, .5, .9]).round(3).tolist()
d["constant_outside_err_q10_50_90_by_top_t"] = cons
out["districts"] = d

rng = np.random.default_rng(0)
idx = np.sort(rng.choice(len(Z), 4000, replace=False))
z = Z[idx].astype(np.float32).transpose(1, 0, 2).reshape(512, -1)
zh = z / np.sqrt((z ** 2).mean(1, keepdims=True)); G = zh @ zh.T / zh.shape[1]; np.fill_diagonal(G, -9)
zc = z - z.mean(1, keepdims=True); zc /= np.sqrt((zc ** 2).mean(1, keepdims=True))
Gc = zc @ zc.T / zc.shape[1]; np.fill_diagonal(Gc, -9)
zc2 = z - z.mean(1, keepdims=True); ev = np.linalg.eigvalsh(zc2 @ zc2.T)[::-1]; ce = np.cumsum(ev) / ev.sum()
out["families"] = {"nn_cosine_z_q10_50_90_99": np.quantile(G.max(1), [.1, .5, .9, .99]).round(3).tolist(),
                   "nn_correlation_z_q10_50_90_99": np.quantile(Gc.max(1), [.1, .5, .9, .99]).round(3).tolist(),
                   "pca_dims_50_80_90_99": [int(np.searchsorted(ce, q)) + 1 for q in (.5, .8, .9, .99)]}
del z, zh, zc, zc2

idx = np.sort(np.random.default_rng(1).choice(len(Z), 12000, replace=False)); L = lab[idx]
z = np.maximum(Z[idx].astype(np.float32), 0)
full = z.mean(2) @ W.T


def logsm(y):
    m = y.max(1, keepdims=True); return y - m - np.log(np.exp(y - m).sum(1, keepdims=True))


st = {}
for q in (0.9, 0.99, 0.999):
    thr = np.quantile(z.transpose(1, 0, 2).reshape(512, -1), q, axis=1)
    s = np.where(z > thr[None, :, None], z, 0).mean(2)
    _, e2, s2 = classtable(s, L)
    y2 = s @ W.T + (full - s @ W.T).mean(0)
    lp, lq = logsm(full), logsm(y2)
    st[q] = {"strong_share_of_activity": float(s.sum() / z.mean(2).sum()),
             "eff_classes_median": float(np.median(e2)),
             "top10_share_median": float(np.median(s2[:, :10].sum(1))),
             "top50_share_median": float(np.median(s2[:, :50].sum(1))),
             "kl_weak_to_mean": float((np.exp(lp) * (lp - lq)).sum(1).mean()),
             "top1_agree": float((full.argmax(1) == y2.argmax(1)).mean())}
out["strong_parts"] = st
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_rz_mapcheck.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
