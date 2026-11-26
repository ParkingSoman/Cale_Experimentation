"""reader, step 2: one photo sheet per pair (JPEG), from fwd.npz and topsets.npz. No model forward.

Rows (8 crops each, one crop per photo, 96 px around the 7 x 7 position, stride 32, 224 px photos):
  1 s top          Small s's strongest rows
  2 p top          Base p's strongest rows
  3 s on, p off    s's strongest rows among rows where p is below its own 90th percentile ("p quiet")
  4 q top          q = Base neuron (not p) whose ON rows (topsets entries > 0, fit+test) hold the most "s on, p quiet" rows
                   (s top 1% and p below its 90th percentile); its top rows (topsets, fit).
Note: Base p's pre-activation is below 0 at > 99% of rows for most pairs, so "p < 0" does not mean "p off".
  5 s bottom       s's lowest pre-activations (negative pole)
  6 p bottom       p's lowest pre-activations
Output: results/pair/reader/sheets/NN_s{s}_p{p}.jpg and sheets.json (q per pair).
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import sparse

from cache import OUT, net

RD = os.path.join(OUT, "reader")


def crop(ph, cache, i, pos):
    if i not in cache:
        cache[i] = Image.fromarray(np.asarray(ph.raw([i])[0]))
    img = cache[i]
    cy, cx = (pos // 7) * 32 + 16, (pos % 7) * 32 + 16
    box = (max(cx - 48, 0), max(cy - 48, 0), min(cx + 48, 224), min(cy + 48, 224))
    return img.crop(box).resize((96, 96))


def rows_to_crops(order, photos):
    seen, out = set(), []
    for r in order:
        i, pos = int(photos[r // 49]), int(r % 49)
        if i in seen:
            continue
        seen.add(i)
        out.append((i, pos))
        if len(out) == 8:
            break
    return out


def s_only_rows(s, p, K):
    quiet = p < np.percentile(p, 90)
    o = np.argsort(-s)
    return o[quiet[o]][:K]


def main():
    d = np.load(os.path.join(RD, "fwd.npz"))
    ts = np.load(os.path.join(OUT, "topsets.npz"))
    pairs, photos = d["pairs"], d["photos"]
    SV, PV, zB = d["SV"].astype(np.float32), d["PV"].astype(np.float32), d["zB"]
    ph = net.Photos("read")
    os.makedirs(os.path.join(RD, "sheets"), exist_ok=True)
    cache, meta = {}, []
    N = SV.shape[0]
    nf = len(ts["fit_photos"]) * 49
    bi = np.concatenate([ts["fit_ids"][3072:], ts["test_ids"][3072:] + nf], 1)  # (4096, 1960)
    on = np.concatenate([ts["fit_vals"][3072:], ts["test_vals"][3072:]], 1).astype(np.float32) > 0  # on-rows only
    Binc = sparse.csr_matrix((on.ravel().astype(np.float32), (bi.ravel(), np.repeat(np.arange(4096), bi.shape[1]))),
                             shape=(N, 4096))
    K = int(N * 0.01)
    for k, (s, p, rf, rw) in enumerate(pairs):
        s, p = int(s), int(p)
        sv, pv = SV[:, k], PV[:, k]
        so = s_only_rows(sv, pv, 100)
        sq = s_only_rows(sv, pv, K)
        cnt = np.asarray(Binc[sq].sum(0)).ravel().astype(float)
        cnt[p] = -1
        q = int(np.argmax(cnt))
        qv = ts["fit_vals"][3072 + q].astype(np.float32)
        qrows = ts["fit_ids"][3072 + q][np.argsort(-qv)]  # strongest on-rows first
        lines = [("s%d top" % s, rows_to_crops(np.argsort(-sv), photos)),
                 ("p%d top" % p, rows_to_crops(np.argsort(-pv), photos)),
                 ("s on, p quiet", rows_to_crops(so, photos)),
                 ("q=b%d top (%d%%)" % (q, 100 * cnt[q] / len(sq)), rows_to_crops(qrows, ts["fit_photos"])),
                 ("s bottom", rows_to_crops(np.argsort(sv), photos)),
                 ("p bottom", rows_to_crops(np.argsort(pv), photos))]
        W, L = 100, 130
        canvas = Image.new("RGB", (L + 8 * W, len(lines) * W + 22), "white")
        dr = ImageDraw.Draw(canvas)
        dr.text((4, 4), f"pair {k}: Small s{s} ~ Base p{p}   R_fire {rf:.2f}  R_write {rw:.2f}", fill="black")
        for row, (lab, cr) in enumerate(lines):
            y = 22 + row * W
            dr.text((4, y + 40), lab, fill="black")
            for col, (i, pos) in enumerate(cr):
                canvas.paste(crop(ph, cache, i, pos), (L + col * W, y))
        fn = os.path.join(RD, "sheets", f"{k:02d}_s{s}_p{p}.jpg")
        canvas.save(fn, quality=80)
        meta.append(dict(k=k, s=s, p=p, rf=float(rf), rw=float(rw), q=q, q_cover=float(cnt[q] / len(sq)),
                         p_cover=float(np.asarray(Binc[sq, p].sum()) / len(sq)), sheet=fn))
        print(k, s, p, q, f"{cnt[q] / len(sq):.2f}", flush=True)
    json.dump(meta, open(os.path.join(RD, "sheets.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
