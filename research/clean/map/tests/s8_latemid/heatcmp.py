"""Heat maps (from out/scan.npz) of the target and chosen candidates on the target's top (or most negative) images.
Usage: python heatcmp.py KEY C1,C6,.. [pos|neg] [K]"""
import sys, os, json, numpy as np
from PIL import Image, ImageDraw
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..", "..", "vision2"))
from convnext import load_images
key = sys.argv[1]; cs = sys.argv[2].split(","); mode = sys.argv[3] if len(sys.argv) > 3 else "pos"; K = int(sys.argv[4]) if len(sys.argv) > 4 else 8
C = json.load(open(os.path.join(H, "..", "..", "s8", "candidates.json")))[key]
D = np.load(os.path.join(H, "out", "scan.npz")); X, _ = load_images(); idx = D["idx"]
tg = [key] + [f"{C[c][0]}_{C[c][1]}" for c in cs]; Zt = D[key].astype(np.float32).reshape(len(idx), -1)
top = np.argsort(-Zt.max(1))[:K] if mode == "pos" else np.argsort(Zt.min(1))[:K]; P = 130
sh = Image.new("RGB", (K * (P + 4), len(tg) * (P + 14)), "white"); d = ImageDraw.Draw(sh)
for r, t in enumerate(tg):
    Z = D[t].astype(np.float32); g = max(abs(Z[top]).max(), 1e-6)
    for j, i in enumerate(top):
        im = np.asarray(X[idx[i]]).astype(np.float32); z = Z[i]; s = max(abs(z).max(), 1e-6)
        hm = np.array(Image.fromarray(((np.clip(z / s, -1, 1) + 1) * 127.5).astype(np.uint8)).resize((224, 224), Image.BILINEAR)).astype(np.float32) / 255 * 2 - 1
        ov = im * .55; ov[..., 0] += 110 * np.clip(hm, 0, 1); ov[..., 2] += 110 * np.clip(-hm, 0, 1)
        sh.paste(Image.fromarray(np.clip(ov, 0, 255).astype(np.uint8)).resize((P, P)), (j * (P + 4), r * (P + 14) + 12))
        d.text((j * (P + 4), r * (P + 14)), (("T" if r == 0 else cs[r - 1]) + f" {t} {z.max():.1f}/{z.min():.1f}"), fill="black")
fp = os.path.join(H, "out", f"heatcmp_{key}_{mode}.png"); sh.save(fp); print(fp)
