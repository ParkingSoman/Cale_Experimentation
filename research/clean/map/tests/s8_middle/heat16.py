"""Full-image heat maps for top-16 and bottom-8 images of one neuron (3000-image set) + where its peak sits.
Usage: python heat16.py B,N [B,N ...]"""
import sys, os, json
import numpy as np
from PIL import Image, ImageDraw
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", "..", "..", "vision2")); sys.path.insert(0, os.path.join(H, "..", "s6"))
from convnext import class_names
import gallery as G
from judge_lib import maps
T = [tuple(map(int, a.split(","))) for a in sys.argv[1:]]; W = {}
for b, n in T: W.setdefault(b, []).append(n)
G._init(); S = G.scan(W); names = [s.split(",")[0] for s in class_names()]; P = 200
for b, n in T:
  vm, pm, vn, pn = S[(b, n)]; R = 56 // 2 ** G._m.stage_of[b]
  order = list(np.argsort(-vm)[:16]) + list(np.argsort(vn)[:8])
  imgs = np.stack([np.asarray(G._X[G._idx[i]]) for i in order]); Z = maps(imgs, [(b, n)])[(b, n)]
  sh = Image.new("RGB", (6 * (P + 4), 4 * (P + 14) + 8), "white"); d = ImageDraw.Draw(sh)
  for j in range(len(order)):
      z = Z[j]; s = max(abs(z).max(), 1e-6)
      hm = np.array(Image.fromarray(((np.clip(z / s, -1, 1) + 1) * 127.5).astype(np.uint8)).resize((224, 224), Image.NEAREST)).astype(np.float32) / 255 * 2 - 1
      ov = imgs[j].astype(np.float32) * .55; ov[..., 0] += 110 * np.clip(hm, 0, 1); ov[..., 2] += 110 * np.clip(-hm, 0, 1)
      r, c = divmod(j, 6); x0, y0 = c * (P + 4), r * (P + 14) + (8 if j >= 16 else 0)
      sh.paste(Image.fromarray(np.clip(ov, 0, 255).astype(np.uint8)).resize((P, P)), (x0, y0 + 12))
      d.text((x0, y0), f"{'+' if j<16 else '-'} {names[G._y[G._idx[order[j]]]][:18]} {z.max():.1f}/{z.min():.1f}", fill="black")
  sh.save(os.path.join(H, "out", f"heat_b{b}_n{n}.png"))
  # where the peak sits (top 150 images by max)
  top = np.argsort(-vm)[:150]; rr, cc = pm[top] // R, pm[top] % R
  edge = ((rr == 0) | (rr == R - 1) | (cc == 0) | (cc == R - 1)).mean(); null_edge = (4 * R - 4) / R ** 2
  bot = np.argsort(vn)[:150]; rn, cn = pn[bot] // R, pn[bot] % R
  edgen = ((rn == 0) | (rn == R - 1) | (cn == 0) | (cn == R - 1)).mean()
  print(b, n, json.dumps(dict(R=R, pos_row_hist=np.bincount(rr, minlength=R).tolist(), pos_col_hist=np.bincount(cc, minlength=R).tolist(),
        pos_edge=round(float(edge), 3), null_edge=round(null_edge, 3), neg_row_hist=np.bincount(rn, minlength=R).tolist(),
        neg_col_hist=np.bincount(cn, minlength=R).tolist(), neg_edge=round(float(edgen), 3))))
