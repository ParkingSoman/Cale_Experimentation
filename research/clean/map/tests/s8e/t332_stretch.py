"""2 n332: contrast-stretched 32-px windows at its top-10 and bottom-10 spots (to see the faint structure)."""
import sys, os, numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "vision2"))
import gallery as G
G._init(); S = G.scan({2: [332]}); vm, pm, vn, pn = S[(2, 332)]
rows = []
for order, P in ((np.argsort(-vm)[:10], pm), (np.argsort(vn)[:10], pn)):
    tiles = []
    for i in order:
        r, c = divmod(int(P[i]), 56); cy, cx = int((r + .5) * 4), int((c + .5) * 4)
        img = np.asarray(G._X[G._idx[i]]).astype(np.float32); t, l = int(np.clip(cy - 16, 0, 192)), int(np.clip(cx - 16, 0, 192))
        w = img[t:t + 32, l:l + 32].mean(2); w = (w - w.min()) / (w.max() - w.min() + 1e-6) * 255
        w = np.repeat(w[..., None], 3, -1); w[16 - (cy - t) % 32 if False else (cy - t), :, 0] = w[cy - t, :, 0]  # no marker
        tiles.append(np.array(Image.fromarray(w.astype(np.uint8)).resize((128, 128), Image.NEAREST)))
        tiles.append(np.full((128, 4, 3), 255, np.uint8))
    rows.append(np.concatenate(tiles, 1)); rows.append(np.full((8, rows[-1].shape[1], 3), 255, np.uint8))
Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t332_stretch.png"))
for name, order, P in (("top", np.argsort(-vm)[:10], pm), ("neg", np.argsort(vn)[:10], pn)):
    print(name, [divmod(int(P[i]), 56) for i in order])
