"""Word-free check of 'smoothness': fine-texture energy of the photo inside each neuron's cell (mean absolute
Laplacian of the grey image, 224 px), at its positive cells (30 band photos), its negative cells (input-only min,
30 band photos), and at random cells of the same photos. Parent cells are 16 px (14 x 14); stage-4 cells 32 px."""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "..")
D = np.load(os.path.join(HERE, "..", "results", "oc_split.npz"))
IXT = np.load(os.path.join(CLEAN, "curves", "cache", "r18", "ix_train.npy"))
X = np.load(os.path.join(CLEAN, "hf", "imagenet", "val_224.npy"), mmap_mode="r")


def lap(r):
    g = np.asarray(X[int(IXT[D["ix"][r]])], np.float32).mean(2)
    return np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])


def cell_energy(L, cell, grid):
    s = 224 // grid; r, c = divmod(int(cell), grid)
    return L[max(r * s - 1, 0):(r + 1) * s - 1, max(c * s - 1, 0):(c + 1) * s - 1].mean()


rng = np.random.default_rng(0)
print(f"{'neuron':6s} {'pos cell':>9s} {'neg cell':>9s} {'random':>8s}   neg/random")
for n in ["p", "k305", "k97", "k23", "k297", "k163", "k275", "c101"]:
    g = 14 if n == "p" else 7
    out = {}
    for side, rows, am in (("pos", D[f"{n}_bpos"], D[f"{n}_amax"]), ("neg", D[f"{n}_bneg"], D[f"{n}_amin"])):
        e, er = [], []
        for r in rows:
            L = lap(r)
            e.append(cell_energy(L, am[r], g)); er.append(cell_energy(L, rng.integers(g * g), g))
        out[side], out[side + "_r"] = np.mean(e), np.mean(er)
    print(f"{n:6s} {out['pos']:9.1f} {out['neg']:9.1f} {out['neg_r']:8.1f}   {out['neg'] / out['neg_r']:.2f}   (pos/random {out['pos'] / out['pos_r']:.2f})")
