"""Children the essay's way, steps 2-3 (LOG "Children, the essay's way"). Mac, numpy only.
Route on one photo = s3:69's exact contribution to each A block-0 neuron at the child cell over the parent's peak,
counted only where the neuron is on, positive part, as shares of the parent's used flow on that photo.
Strong photos = parent peak at least half of its own max. Photos are grouped by route vectors (cosine k-means,
k = 2..8). For each k: group sizes, animal share, split-half agreement, the neurons that carry each group.
Sheets for the agent: results/route_k{k}.png (12 crops per group around the parent's peak cell).
Saved: results/oc_route.npz. Code does not pick members; an agent reads and proposes.
"""
import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
CLEAN = os.path.join(HERE, "..", "..")
S1 = np.load(os.path.join(R, "oc_step1.npz"))
ppk = S1["ppk"].astype(np.float32)
strong = np.where(ppk >= .5 * ppk.max())[0]
C = np.clip(S1["A_c69"][strong].astype(np.float32), 0, None) * (S1["A_pre"][strong].astype(np.float32) > 0)
tot = C.sum(1, keepdims=True)
keep = tot[:, 0] > 0
strong, C, tot = strong[keep], C[keep], tot[keep]
SH = C / tot                                                    # shares, photos x 512
lab = S1["lab"][strong]  # labels: record-only (split WITHDRAWN 2026-11-23, LOG "Withdrawn")
animal = lab < 398  # labels: record-only
parg = S1["parg"][strong].astype(int)
print(f"strong photos: {len(strong)} (half of max {ppk.max():.2f}); animal share {animal.mean():.2f}")


def kmeans(F, k, seed=0, it=100):
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    best = None
    for s in range(5):
        rng = np.random.default_rng(seed + s)
        Cn = F[rng.choice(len(F), k, replace=False)]
        for _ in range(it):
            a = (F @ Cn.T).argmax(1)
            new = np.stack([F[a == g].mean(0) if (a == g).any() else Cn[g] for g in range(k)])
            new /= np.linalg.norm(new, axis=1, keepdims=True) + 1e-9
            if np.allclose(new, Cn):
                break
            Cn = new
        score = (F * Cn[a]).sum()
        if best is None or score > best[0]:
            best = (score, a, Cn)
    return best[1], best[2]


def half_agree(k):
    """Cluster each half alone, assign the other half by nearest centroid of each run; agreement of labels after
    matching groups by cosine (Hungarian not needed for small k: greedy)."""
    h0, h1 = np.arange(0, len(SH), 2), np.arange(1, len(SH), 2)
    a0, C0 = kmeans(SH[h0], k)
    a1, C1 = kmeans(SH[h1], k)
    F = SH / (np.linalg.norm(SH, axis=1, keepdims=True) + 1e-9)
    l0, l1 = (F @ C0.T).argmax(1), (F @ C1.T).argmax(1)
    sim = C0 @ C1.T
    m = {}
    for _ in range(k):
        i, j = np.unravel_index(np.argmax(sim), sim.shape)
        m[j] = i; sim[i, :] = -9; sim[:, j] = -9
    return float((l0 == np.array([m[x] for x in l1])).mean())


IXT = np.load(os.path.join(CLEAN, "curves", "cache", "r18", "ix_train.npy"))
X = np.load(os.path.join(CLEAN, "hf", "imagenet", "val_224.npy"), mmap_mode="r")
T = 96


def crop(n_row, cell):
    raw = np.asarray(X[int(IXT[S1["ix"][n_row]])])
    rr, cc = divmod(int(cell), 14); c = 224 / 14
    cx, cy = (cc + .5) * c, (rr + .5) * c; Z = 96
    l, t = int(min(max(cx - Z / 2, 0), 224 - Z)), int(min(max(cy - Z / 2, 0), 224 - Z))
    im = Image.fromarray(raw).crop((l, t, l + Z, t + Z)).resize((T, T))
    k = T / Z
    ImageDraw.Draw(im).rectangle(((cx - c / 2 - l) * k, (cy - c / 2 - t) * k, (cx + c / 2 - l) * k, (cy + c / 2 - t) * k), outline=(255, 255, 0))
    return im


out = {"strong": strong, "SH": SH.astype(np.float16), "animal": animal}
print(f"\n{'k':>2} | half-agree | groups: size / animal share / top carriers (neuron:mean share)")
for k in range(2, 9):
    a, Cn = kmeans(SH, k)
    ag = half_agree(k)
    out[f"a{k}"] = a
    print(f"{k:2d} | {ag:.2f}")
    rows = []
    for g in range(k):
        m = a == g
        ms = SH[m].mean(0); top = np.argsort(-ms)[:6]
        print(f"     g{g}: n={m.sum():3d} animal {animal[m].mean():.2f} | " + " ".join(f"n{j}:{ms[j]:.2f}" for j in top))
        idx = np.where(m)[0]
        idx = idx[np.argsort(-ppk[strong[idx]])][:12]
        rows.append((g, m.sum(), animal[m].mean(), top, ms, idx))
    if k <= 6:
        img = Image.new("RGB", (12 * T, k * (T + 16)), "white"); dr = ImageDraw.Draw(img)
        for i, (g, n, an, top, ms, idx) in enumerate(rows):
            y = i * (T + 16)
            dr.text((2, y), f"group {g}: {n} photos, animal {an:.2f}; carriers " + ", ".join(f"n{j} {ms[j]:.2f}" for j in top[:4]), fill="black")
            for q, r in enumerate(idx):
                img.paste(crop(strong[r], parg[r]), (q * T, y + 14))
        img.save(os.path.join(R, f"route_k{k}.png"))

tot_share = SH.mean(0); top = np.argsort(-tot_share)[:30]
print("\nneurons by mean share of the parent's used flow on strong photos:")
print(" ".join(f"n{j}:{tot_share[j]:.3f}" for j in top))
print("share held by top 6 / 12 / 20 / 40:", [round(float(np.sort(tot_share)[::-1][:n].sum()), 2) for n in (6, 12, 20, 40)])
np.savez(os.path.join(R, "oc_route.npz"), **out)
