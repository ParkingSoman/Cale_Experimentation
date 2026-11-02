"""Zoomed top spots with heat overlay: python fig_zoom.py BLOCK NEURON [k] -> map/tests/zoom_b{b}_n{n}.png"""
import sys, os, numpy as np
from PIL import Image, ImageDraw
from early_lib import *
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vision2"))
import gallery as G
b, n = int(sys.argv[1]), int(sys.argv[2]); k = int(sys.argv[3]) if len(sys.argv) > 3 else 16
vm, pm, vn, pn = G.scan({b: [n]})[(b, n)]; R = 56 // 2 ** G._m.stage_of[b]; s = 224 // R; cs = 96
out = Image.new("RGB", (8 * 200, (k // 8) * 2 * 100 * 2 // 2 * 2), "white")
rows = []
for side, ix, P in (("top", np.argsort(-vm)[:k], pm), ("neg", np.argsort(vn)[:8], pn)):
    imgs = np.stack([np.asarray(G._X[G._idx[i]]) for i in ix]); Z = zmaps(b, n, imgs)
    for j, i in enumerate(ix):
        r, c = divmod(int(P[i]), R); cy, cx = int((r + .5) * s), int((c + .5) * s)
        t = int(np.clip(cy - cs / 2, 0, 224 - cs)); l = int(np.clip(cx - cs / 2, 0, 224 - cs))
        im = imgs[j].astype(np.float32); z = Z[j]; sc = max(abs(z).max(), 1e-6)
        hm = np.array(Image.fromarray(((np.clip(z / sc, -1, 1) + 1) * 127.5).astype(np.uint8)).resize((224, 224), Image.NEAREST)).astype(np.float32) / 255 * 2 - 1
        ov = im * .6; ov[..., 0] += 100 * np.clip(hm, 0, 1); ov[..., 2] += 100 * np.clip(-hm, 0, 1)
        a = Image.fromarray(imgs[j][t:t + cs, l:l + cs]).resize((96, 96)); bimg = Image.fromarray(np.clip(ov, 0, 255).astype(np.uint8)[t:t + cs, l:l + cs]).resize((96, 96))
        rows.append((side, a, bimg))
W = 8 * 200; H = ((len(rows) + 7) // 8) * 110 + 10; out = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(out)
for q, (side, a, bimg) in enumerate(rows):
    x, y = (q % 8) * 200, (q // 8) * 110 + 5; out.paste(a, (x, y)); out.paste(bimg, (x + 98, y)); d.text((x, y + 97), side, fill="red" if side == "neg" else "green")
fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"zoom_b{b}_n{n}.png"); out.save(fp); print(fp)
