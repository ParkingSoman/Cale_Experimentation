"""Close-up sheet from out/scan.npz: top-k / most-negative spots for one neuron; crop + heat overlay.
Usage: python closeup.py B_N [cs] [k] [other B_N to overlay instead]"""
import sys, os, numpy as np
from PIL import Image, ImageDraw
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..", "..", "vision2"))
from convnext import class_names, load_images
key = sys.argv[1]; cs = int(sys.argv[2]) if len(sys.argv) > 2 else 80; k = int(sys.argv[3]) if len(sys.argv) > 3 else 16
ov_key = sys.argv[4] if len(sys.argv) > 4 else key
D = np.load(os.path.join(H, "out", "scan.npz")); Z = D[key].astype(np.float32); ZO = D[ov_key].astype(np.float32); idx = D["idx"]; y = D["y"]
X, _ = load_images(); names = [s.split(",")[0] for s in class_names()]; P = 140
for tag in ("pos", "neg"):
    v = Z.reshape(len(Z), -1); order = np.argsort(-v.max(1))[:k] if tag == "pos" else np.argsort(v.min(1))[:k]
    sh = Image.new("RGB", (4 * (2 * P + 10), ((k + 3) // 4) * (P + 16)), "white"); d = ImageDraw.Draw(sh)
    for j, i in enumerate(order):
        pos = v[i].argmax() if tag == "pos" else v[i].argmin(); r, c = divmod(int(pos), 14); s = 16
        cy, cx = (r + .5) * s, (c + .5) * s; t = int(np.clip(cy - cs / 2, 0, 224 - cs)); l = int(np.clip(cx - cs / 2, 0, 224 - cs))
        im = np.asarray(X[idx[i]]); z = ZO[i]; sc = max(abs(z).max(), 1e-6)
        hm = np.array(Image.fromarray(((np.clip(z / sc, -1, 1) + 1) * 127.5).astype(np.uint8)).resize((224, 224), Image.BILINEAR)).astype(np.float32) / 255 * 2 - 1
        ov = im.astype(np.float32) * .6; ov[..., 0] += 100 * np.clip(hm, 0, 1); ov[..., 2] += 100 * np.clip(-hm, 0, 1)
        a = Image.fromarray(im[t:t + cs, l:l + cs]).resize((P, P)); o = Image.fromarray(np.clip(ov, 0, 255).astype(np.uint8)).resize((P, P))
        x0, y0 = (j % 4) * (2 * P + 10), (j // 4) * (P + 16); sh.paste(a, (x0, y0)); sh.paste(o, (x0 + P, y0))
        d.text((x0, y0 + P + 1), f"{names[y[i]][:18]} {v[i].max() if tag == 'pos' else v[i].min():.1f}", fill="black")
    fp = os.path.join(H, "out", f"close_{key}_{tag}" + (f"_ov{ov_key}" if ov_key != key else "") + ".png"); sh.save(fp); print(fp)
