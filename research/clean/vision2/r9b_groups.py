"""R9b: label-free groups (k-means on the model's final feature) in place of classes. Rules: VISION-LOG.md R9b."""
import json
import numpy as np, mlx.core as mx
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from convnext import *
from acts import get_acts

X, y = load_images(); names = [n.split(",")[0] for n in class_names()]
amax, _ = get_acts(); ana, _ = split(y); ya = y[ana]; rng = np.random.default_rng(0)
ff = os.path.join(HF, "imagenet", "feat_tiny_ana20k.npy")
if not os.path.exists(ff):
    m = load_convnext(); F = []
    for i in range(0, len(ana), 100):
        s = m.stream(mx.array(norm_img(np.asarray(X[ana[i:i + 100]])))); F.append(np.array(m.head_norm(s.mean((1, 2)))))
    np.save(ff, np.concatenate(F))
F = np.load(ff); F = F / np.linalg.norm(F, axis=1, keepdims=True)


def terr_groups(g, k):
    out = {}
    for b, a in amax.items():
        cm = np.zeros((k, a.shape[1]), np.float32); np.add.at(cm, g, a); cm /= np.bincount(g, minlength=k)[:, None]
        med = np.median(cm, 0); out[b] = (cm - med) / (cm.max(0) - med + 1e-9) >= .5
    return out


def nested(kid, par):
    k, p = kid.astype(np.float32), par.astype(np.float32); inter = k.T @ p; nk, br = k.sum(0), p.sum(0)
    cont = inter / np.maximum(nk[:, None], 1); jac = inter / np.maximum(nk[:, None] + br[None] - inter, 1)
    ok = (cont >= .8) & (br[None] > nk[:, None]); best = np.where(ok.any(1), np.argmax(np.where(ok, jac, -1), 1), -1)
    return ok.any(1), best, cont


TC = terr_groups(ya, 1000); out = {}
for k in (1000, 300):
    g = KMeans(k, n_init=1, random_state=0).fit_predict(F); T = terr_groups(g, k); o = {"nmi": float(nmi(ya, g)), "scale": {}}
    print(f"\n== k = {k}: NMI with classes {o['nmi']:.2f}")
    for b in range(12, 18):
        has, best, cont = nested(T[b], T[b - 1])
        sh = T[b - 1][rng.permutation(k)]; hs, _, _ = nested(T[b], sh)
        o["scale"][b] = {"median_breadth": float(np.median(T[b].sum(0))), "has": float(has.mean()), "null": float(hs.mean())}
        print(f"block {b} <- {b-1}: median breadth {np.median(T[b].sum(0)):.0f} groups | nested parent {has.mean():.2f} | shuffled null {hs.mean():.2f}")
    # agreement with class level, blocks 16->17
    hC, bC, _ = nested(TC[17], TC[16]); has, best, _ = nested(T[17], T[16]); both = has & hC
    o["same_top_parent_17"] = float((best[both] == bC[both]).mean()); o["n_both"] = int(both.sum())
    print(f"  same top parent as class level (16->17): {o['same_top_parent_17']:.2f} of {both.sum()} children")
    b, n, rows = 17, 2283, []
    while True:
        rows.append((b, n, int(T[b][:, n].sum()), int(TC[b][:, n].sum())))
        if b - 1 < 11: break
        h, bb, _ = nested(T[b][:, [n]], T[b - 1])
        if not h[0]: break
        b, n = b - 1, int(bb[0])
    o["chain"] = rows
    print("  chain from b17 n2283:", " <- ".join(f"b{b} n{n} ({ng} groups; {nc} cls)" for b, n, ng, nc in rows))
    for b, n, _, _ in rows:
        cl = np.where(TC[b][:, n])[0]; print(f"    b{b} n{n} class territory:", [names[c] for c in cl[:10]])
    out[k] = o
json.dump(out, open(os.path.join(RES, "vision-r9b-groups.json"), "w"), indent=1, default=str)
