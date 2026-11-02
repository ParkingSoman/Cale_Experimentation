"""R11: is the territory tree true for ALL neurons? Rules: VISION-LOG.md R11."""
import json
import numpy as np
from sklearn.cluster import KMeans
from convnext import *
from acts import get_acts

amax, _ = get_acts(); X, y = load_images(); ana, _ = split(y)
F = np.load(os.path.join(HF, "imagenet", "feat_tiny_ana20k.npy")); F = F / np.linalg.norm(F, axis=1, keepdims=True)
K = 1000; g = KMeans(K, n_init=1, random_state=0).fit_predict(F); rng = np.random.default_rng(1)
T = {}
for b, a in amax.items():
    cm = np.zeros((K, a.shape[1]), np.float32); np.add.at(cm, g, a); cm /= np.bincount(g, minlength=K)[:, None]
    med = np.median(cm, 0); T[b] = ((cm - med) / (cm.max(0) - med + 1e-9) >= .5).astype(np.float32)


def cont(kid, par):
    return (kid.T @ par) / np.maximum(kid.sum(0)[:, None], 1)


def single(kid, par):
    c = cont(kid, par); ok = (c >= .8) & (par.sum(0)[None] > kid.sum(0)[:, None]); return ok.any(1), np.where(par.sum(0)[None] > kid.sum(0)[:, None], c, 0).max(1)


def pair(kid, par):
    """greedy union of 2: best single (any breadth), then best addition. -> containment of the union."""
    c = cont(kid, par); p1 = c.argmax(1); rest = kid * (1 - par[:, p1])          # child groups not covered by p1
    add = (rest.T @ par) / np.maximum(kid.sum(0)[:, None], 1); return c.max(1) + add.max(1) >= .8


def classify(b, shuffle=False):
    P = {bb: (T[bb][rng.permutation(K)] if shuffle else T[bb]) for bb in range(11, b)}
    kid = T[b]; n = kid.shape[1]; lab = np.full(n, "", object)
    s1, best1 = single(kid, P[b - 1]); lab[s1] = "1 parent b-1"
    s2 = np.zeros(n, bool)
    for bb in range(11, b - 1): s2 |= single(kid, P[bb])[0]
    lab[(lab == "") & s2] = "2 parent earlier"
    s3 = pair(kid, P[b - 1]); lab[(lab == "") & s3] = "3 two parents"
    root = kid.sum(0) > np.median(T[b - 1].sum(0)); lab[(lab == "") & root] = "4 root"
    lab[lab == ""] = "5 orphan"
    return lab, best1


out = {}
cats = ["1 parent b-1", "2 parent earlier", "3 two parents", "4 root", "5 orphan"]
print("block | " + " | ".join(cats) + "   (shuffled null in brackets)")
for b in range(12, 18):
    lab, best1 = classify(b); labn, _ = classify(b, True)
    fr = {c: float((lab == c).mean()) for c in cats}; frn = {c: float((labn == c).mean()) for c in cats}
    orph = lab == "5 orphan"; nm = float(((best1 >= .6) & orph).sum() / max(orph.sum(), 1))
    out[b] = {"real": fr, "null": frn, "orphan_near_miss_.6": nm, "orphan_median_breadth": float(np.median(T[b].sum(0)[orph])) if orph.any() else None,
              "all_median_breadth": float(np.median(T[b].sum(0)))}
    print(f"{b:5d} | " + " | ".join(f"{fr[c]:.2f} ({frn[c]:.2f})" for c in cats),
          f"| orphans near-miss {nm:.2f}, orphan median breadth {out[b]['orphan_median_breadth']} vs all {out[b]['all_median_breadth']}")
json.dump(out, open(os.path.join(RES, "vision-r11-all.json"), "w"), indent=1)
