"""Real-image window and tile edits around a target's peak cell (top-N images of the 20k analysis set)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import last_lib as L
from judge_lib import maps


def top_images(b, n, k=20, neg=False, classes=None):
    X, y, ana = L.data(); v = L.amin(b, n) if neg else L.amax(b, n); ya = y[ana]
    o = np.argsort(v if neg else -v)
    if classes is not None: o = [i for i in o if ya[i] in classes]
    idx = ana[np.asarray(o[:k])]; return np.stack([np.asarray(X[i]) for i in idx]), idx


def peak_px(z, neg=False):
    """centre pixel of the peak cell of map z (R,R)."""
    R = z.shape[0]; r, c = np.unravel_index(z.argmin() if neg else z.argmax(), z.shape); s = 224 / R
    return int((r + .5) * s), int((c + .5) * s)


def tile(img, cy, cx, p=64):
    y0 = int(np.clip(cy - p // 2, 0, 224 - p)); x0 = int(np.clip(cx - p // 2, 0, 224 - p)); patch = img[y0:y0 + p, x0:x0 + p]
    k = 224 // p + 1; return np.tile(patch, (k, k, 1))[:224, :224]


def window(img, cy, cx, w=96, gray=128):
    out = np.full_like(img, gray); y0 = int(np.clip(cy - w // 2, 0, 224 - w)); x0 = int(np.clip(cx - w // 2, 0, 224 - w))
    out[y0:y0 + w, x0:x0 + w] = img[y0:y0 + w, x0:x0 + w]; return out
