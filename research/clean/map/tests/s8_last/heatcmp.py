"""Heat maps of the target and chosen candidates on the target's top images (20k set). Usage: python heatcmp.py KEY C1,C6,..."""
import sys, os, json, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from judge_lib import maps
import gallery as G
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
key = sys.argv[1]; C = json.load(open(os.path.join(D, "s8", "candidates.json")))[key]; T = tuple(map(int, key.split("_")))
cs = sys.argv[2].split(","); tg = [T] + [tuple(C[c]) for c in cs]; K = 6
G._init(); S = G.scan({T[0]: [T[1]]}); vm = S[T][0]; top = np.argsort(-vm)[:K]
imgs = np.stack([np.asarray(G._X[G._idx[i]]) for i in top]); Z = maps(imgs, tg); P = 150
sh = Image.new("RGB", (K * (P + 4), len(tg) * (P + 14)), "white"); d = ImageDraw.Draw(sh)
for r, t in enumerate(tg):
    mxv = max(abs(Z[t]).max(), 1e-6)
    for j in range(K):
        z = Z[t][j]; s = max(abs(z).max(), 1e-6)
        hm = np.array(Image.fromarray(((np.clip(z / s, -1, 1) + 1) * 127.5).astype(np.uint8)).resize((224, 224), Image.NEAREST)).astype(np.float32) / 255 * 2 - 1
        ov = imgs[j].astype(np.float32) * .55; ov[..., 0] += 110 * np.clip(hm, 0, 1); ov[..., 2] += 110 * np.clip(-hm, 0, 1)
        sh.paste(Image.fromarray(np.clip(ov, 0, 255).astype(np.uint8)).resize((P, P)), (j * (P + 4), r * (P + 14) + 12))
        d.text((j * (P + 4), r * (P + 14)), (("T" if r == 0 else cs[r - 1]) + f" b{t[0]}n{t[1]} max {z.max():.1f}"), fill="black")
fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", f"heatcmp_{key}.png"); sh.save(fp); print(fp)
