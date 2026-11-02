"""Study tools for fine (original 3b) and coarse (crushed) neurons.
A 'neuron set' is a function ins -> (B, n, 28, 28) of the four 3b inputs (pre-ReLU values)."""
import numpy as np, torch
from PIL import Image
from inception import *
from stimuli import arc, line, circle


def fine_fn(m, idx=None):
    def f(ins):
        z = pre_3b(m, ins[0])
        return z if idx is None else z[:, idx]
    return f


def coarse_fn(cr):
    return lambda ins: cr.pre(ins)


def tuning(m, fn, radii=(20, 40, 60, 100), th=np.arange(0, 360, 10)):
    """Response (post-ReLU, max over centre 5x5) to arcs: dict radius -> (n_theta, n_neurons)."""
    out = {}
    with torch.no_grad():
        for r in radii:
            S = np.stack([arc(t, radius=r) for t in th])
            out[r] = F.relu(fn(inputs_3b(m, to_3a(m, prep(S)))))[:, :, 12:17, 12:17].amax((2, 3)).cpu().numpy()
    return out


def top_spots(m, fn, X, idx, n_keep=12, bs=100, neg=False):
    """Strongest (or most negative pre-ReLU) positions per neuron over images idx: list of (value, img, i, j)."""
    best = None
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]; z = fn(inputs_3b(m, to_3a(m, prep(X[b]))))
            z = -z if neg else z
            v, p = z.flatten(2).max(2)            # best position per image per neuron
            v, p = v.cpu().numpy(), p.cpu().numpy()
            cand = [(v[:, k], b, p[:, k]) for k in range(z.shape[1])]
            if best is None: best = [[] for _ in range(z.shape[1])]
            for k, (vk, bk, pk) in enumerate(cand):
                o = np.argsort(-vk)[:n_keep]
                best[k] = sorted(best[k] + [(float(vk[i]), int(bk[i]), int(pk[i]) // 28, int(pk[i]) % 28) for i in o], reverse=True)[:n_keep]
    return best


def crop(X, img, i, j, size=72):
    cy, cx = 8 * i + 4, 8 * j + 4; h = size // 2
    pad = np.pad(np.asarray(X[img]), ((h, h), (h, h), (0, 0)), mode="constant", constant_values=255)
    return pad[cy:cy + size, cx:cx + size]


def sheet(X, spots, path, size=72, labels=None):
    """One row per neuron: its top crops."""
    rows = len(spots); cols = max(len(s) for s in spots)
    S = np.full((rows * (size + 4), cols * (size + 4), 3), 255, np.uint8)
    for r, sp in enumerate(spots):
        for c, (v, img, i, j) in enumerate(sp):
            S[r * (size + 4):r * (size + 4) + size, c * (size + 4):c * (size + 4) + size] = crop(X, img, i, j, size)
    Image.fromarray(S).save(path)
    return path
