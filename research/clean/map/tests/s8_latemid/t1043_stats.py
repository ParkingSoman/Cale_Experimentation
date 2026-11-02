"""14_1043: what local content goes with high values in the border ring? For every cell of the 3000-image set:
detail (mean abs gradient in the 32x32 window), brightness, saturation. Compare cells with T > .3 max vs all cells,
separately for the border ring and the interior. Also detail of the whole image (close-up vs scene)."""
import sys, os, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..", "..", "vision2"))
from convnext import load_images
D = np.load(os.path.join(H, "out", "scan.npz")); X, _ = load_images(); idx = D["idx"]; Z = D["14_1043"].astype(np.float32)
I = np.stack([np.asarray(X[i]) for i in idx]).astype(np.float32); g = I.mean(-1)
gr = np.abs(np.diff(g, axis=1))[:, :, :-1] + np.abs(np.diff(g, axis=2))[:, :-1, :]; gr = np.pad(gr, ((0, 0), (0, 1), (0, 1)))
det = gr.reshape(-1, 14, 16, 14, 16).mean((2, 4)); bri = g.reshape(-1, 14, 16, 14, 16).mean((2, 4))
sat = (I.max(-1) - I.min(-1)).reshape(-1, 14, 16, 14, 16).mean((2, 4))
from scipy.ndimage import uniform_filter
det2 = uniform_filter(det, size=(1, 3, 3), mode="nearest")
ring = np.zeros((14, 14), bool); ring[0] = ring[-1] = ring[:, 0] = ring[:, -1] = True
on = Z > .3 * Z.max(); imgdet = det.mean((1, 2))
for nm, m in (("ring", ring), ("interior", ~ring)):
    o = on[:, m]; print(f"{nm}: on cells {o.sum()}")
    for lab, a in (("detail 16px", det), ("detail 48px", det2), ("brightness", bri), ("saturation", sat)):
        v = a[:, m]; print(f"   {lab:12s} on {np.median(v[o]):7.2f}   all {np.median(v):7.2f}   neg(T<.3min) {np.median(v[(Z < .3 * Z.min())[:, m]]):7.2f}")
top = np.argsort(-Z.reshape(len(Z), -1).max(1))[:100]
print(f"whole-image detail: top-100 images {np.median(imgdet[top]):.2f}  all {np.median(imgdet):.2f}")
