"""Close-up sheet for one neuron: top-k and most-negative spots on the 3000-image set. Each cell = crop (cs px)
around the spot, upscaled, plus the same crop with the neuron's heat overlay. Usage: python closeup.py B N cs [k]"""
import sys, os
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "vision2"))
from convnext import class_names
import gallery as G
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_lib import maps
b, n, cs = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); k = int(sys.argv[4]) if len(sys.argv) > 4 else 12
G._init(); S = G.scan({b: [n]}); vm, pm, vn, pn = S[(b, n)]; R = 56 // 2 ** G._m.stage_of[b]
names = [s.split(",")[0] for s in class_names()]; P = 150
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
for tag, order, pos in (("pos", np.argsort(-vm)[:k], pm), ("neg", np.argsort(vn)[:k], pn)):
    imgs = np.stack([np.asarray(G._X[G._idx[i]]) for i in order]); Z = maps(imgs, [(b, n)])[(b, n)]
    sh = Image.new("RGB", (4 * (2 * P + 10), ((k + 3) // 4) * (P + 16)), "white"); d = ImageDraw.Draw(sh)
    for j, i in enumerate(order):
        r, c = divmod(int(pos[i]), R); s = 224 / R; cy, cx = (r + .5) * s, (c + .5) * s
        t = int(np.clip(cy - cs / 2, 0, 224 - cs)); l = int(np.clip(cx - cs / 2, 0, 224 - cs))
        im = imgs[j].astype(np.float32); z = Z[j]; sc = max(abs(z).max(), 1e-6)
        hm = np.array(Image.fromarray(((np.clip(z / sc, -1, 1) + 1) * 127.5).astype(np.uint8)).resize((224, 224), Image.NEAREST)).astype(np.float32) / 255 * 2 - 1
        ov = im * .6; ov[..., 0] += 100 * np.clip(hm, 0, 1); ov[..., 2] += 100 * np.clip(-hm, 0, 1)
        a = Image.fromarray(imgs[j][t:t + cs, l:l + cs]).resize((P, P), Image.NEAREST); o = Image.fromarray(np.clip(ov, 0, 255).astype(np.uint8)[t:t + cs, l:l + cs]).resize((P, P), Image.NEAREST)
        x0, y0 = (j % 4) * (2 * P + 10), (j // 4) * (P + 16); sh.paste(a, (x0, y0)); sh.paste(o, (x0 + P, y0))
        dd = ImageDraw.Draw(sh); dd.rectangle([x0 + P // 2 - 3, y0 + P // 2 - 3, x0 + P // 2 + 3, y0 + P // 2 + 3], outline="yellow")
        d.text((x0, y0 + P + 1), f"{names[G._y[G._idx[i]]][:16]} {z.max() if tag=='pos' else z.min():.1f}", fill="black")
    fp = os.path.join(OUT, f"close_b{b}_n{n}_{tag}.png"); sh.save(fp); print(fp)
