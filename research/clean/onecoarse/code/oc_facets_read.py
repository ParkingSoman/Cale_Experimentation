"""Children the essay's way, step 6 on the Mac: n101's facets and P3 (LOG "Children, the essay's way").
n101's strong photos = input peak at least half of its own max. Route = positive per-channel contributions at
the peak cell, as shares. Grouped by cosine k-means (k = 2..6, same code as oc_route.py), with split-half agreement.
P3: each of n101's strong photos is also given the parent's route (s3:69 into A block 0, oc_step1.npz) and assigned
to the nearest parent group centroid (k = 4 and 5 from oc_route.npz). Cross-table: n101 facet x parent group.
Sheet: results/facets_k{k}.png. Saved: results/oc_facets_read.npz
"""
import os

import numpy as np
from PIL import Image, ImageDraw

import oc_route as OR                                          # reuses kmeans, crop data (runs oc_route once)

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
Fc = np.load(os.path.join(R, "oc_facets.npz"))
S1 = OR.S1
assert (Fc["ix"] == S1["ix"]).all()
pk = Fc["pk"].astype(np.float32)
st = np.where(pk >= .5 * pk.max())[0]
SP = np.clip(Fc["split"][st].astype(np.float32), 0, None)
SP = SP / SP.sum(1, keepdims=True)
animal = S1["lab"][st] < 398  # labels: record-only (split WITHDRAWN 2026-11-23, LOG "Withdrawn")
print(f"\nn101 strong photos: {len(st)} (half of max {pk.max():.2f}); animal {animal.mean():.2f}")
s = np.sort(SP, 1)[:, ::-1]
print("n101 per photo: top channel share median", round(float(np.median(s[:, 0])), 3), "| channels to reach half: median", int(np.median((np.cumsum(s, 1) < .5).sum(1) + 1)))

# parent route on n101's strong photos, assigned to parent groups
C = np.clip(S1["A_c69"][st].astype(np.float32), 0, None) * (S1["A_pre"][st].astype(np.float32) > 0)
ok = C.sum(1) > 0
PR = C / np.maximum(C.sum(1, keepdims=True), 1e-9)
PRn = PR / (np.linalg.norm(PR, axis=1, keepdims=True) + 1e-9)
SHp = OR.SH / (np.linalg.norm(OR.SH, axis=1, keepdims=True) + 1e-9)
parent_strong = set(OR.strong.tolist())
print("n101 strong photos that are also parent-strong:", sum(int(i) in parent_strong for i in st), "of", len(st))

out = {"st": st}
for k in range(2, 7):
    a = OR.kmeans(SP, k)[0]
    # split-half agreement, same method as oc_route.half_agree but on SP
    h0, h1 = np.arange(0, len(SP), 2), np.arange(1, len(SP), 2)
    _, C0 = OR.kmeans(SP[h0], k); _, C1 = OR.kmeans(SP[h1], k)
    Fn = SP / (np.linalg.norm(SP, axis=1, keepdims=True) + 1e-9)
    l0, l1 = (Fn @ C0.T).argmax(1), (Fn @ C1.T).argmax(1)
    sim = C0 @ C1.T; m = {}
    for _ in range(k):
        i, j = np.unravel_index(np.argmax(sim), sim.shape); m[j] = i; sim[i, :] = -9; sim[:, j] = -9
    ag = float((l0 == np.array([m[x] for x in l1])).mean())
    out[f"a{k}"] = a
    print(f"\nk={k} n101 facets, half-agree {ag:.2f}")
    for g in range(k):
        mm = a == g; ms = SP[mm].mean(0); top = np.argsort(-ms)[:6]
        print(f"   f{g}: n={mm.sum():3d} animal {animal[mm].mean():.2f} | " + " ".join(f"s3:{c}:{ms[c]:.3f}" for c in top))
    if k in (4, 5):
        for kp in (4, 5):
            pa = OR.__dict__.get("_dummy")  # placeholder to keep names clear
            ap = np.load(os.path.join(R, "oc_route.npz"))[f"a{kp}"]
            cent = np.stack([SHp[ap == g].mean(0) for g in range(kp)])
            cent /= np.linalg.norm(cent, axis=1, keepdims=True)
            pg = (PRn @ cent.T).argmax(1)
            tab = np.zeros((k, kp), int)
            for f, p in zip(a[ok], pg[ok]):
                tab[f, p] += 1
            purity = tab.max(1) / np.maximum(tab.sum(1), 1)
            # chance purity: shuffle facet labels
            rng = np.random.default_rng(0); ch = []
            for _ in range(200):
                sa = rng.permutation(a[ok]); t = np.zeros((k, kp), int)
                for f, p in zip(sa, pg[ok]): t[f, p] += 1
                ch.append((t.max(1) / np.maximum(t.sum(1), 1)).mean())
            print(f"   P3 vs parent groups k={kp}: facet purity " + " ".join(f"{x:.2f}" for x in purity) + f" | mean {purity.mean():.2f}, shuffled 99th pct {np.quantile(ch, .99):.2f}")
            print("      table (rows n101 facets, cols parent groups):", tab.tolist())
    if k <= 5:
        T = OR.T
        img = Image.new("RGB", (12 * T, k * (T + 16)), "white"); dr = ImageDraw.Draw(img)
        arg = Fc["arg"][st].astype(int)
        for g in range(k):
            idx = np.where(a == g)[0]; idx = idx[np.argsort(-pk[st[idx]])][:12]
            y = g * (T + 16)
            ms = SP[a == g].mean(0); top = np.argsort(-ms)[:4]
            dr.text((2, y), f"n101 facet {g}: {int((a == g).sum())} photos, animal {animal[a == g].mean():.2f}; channels " + ", ".join(f"s3:{c} {ms[c]:.3f}" for c in top), fill="black")
            for q, r in enumerate(idx):
                rr, cc = divmod(int(arg[r]), 7)
                img.paste(OR.crop(st[r], (rr * 2) * 14 + cc * 2), (q * T, y + 14))   # 7x7 cell -> 14x14 cell
        img.save(os.path.join(R, f"facets_k{k}.png"))
np.savez(os.path.join(R, "oc_facets_read.npz"), **out)
