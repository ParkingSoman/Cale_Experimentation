"""Read, one firing at a time: one sheet per subcircuit for the reader agents (LOG "Read, one firing at a time").
Mac, numpy + PIL. Row 0 draws DEV firings only. The counterpart rows draw each neuron's own top 12 of the 19,000
photos, which can include n101's CHECK photos (this was noticed after the CHECK scores were computed).

Rows of results/sub_<pole>_k<k>.png:
  row 0      n101 on the subcircuit's DEV firings: intersection images (colour mean and edge mean of a 4-cell window
             centred on n101's cell, essay footnote 4), then 8 crops (largest n101 input first)
  rows 1-3   the subcircuit's top 3 counterparts (A stage-4 block-0 neurons), each on ITS OWN 12 strongest of the 19,000
             held + test photos at its own peak cell: intersection images, then 8 crops
Usage: python oc_subsheet.py [pos|neg]
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
CLEAN = os.path.join(HERE, "..", "..")
X = np.load(os.path.join(CLEAN, "hf", "imagenet", "val_224.npy"), mmap_mode="r")
IXT = np.load(os.path.join(CLEAN, "curves", "cache", "r18", "ix_train.npy"))
RA = np.load(os.path.join(CLEAN, "neuron", "results", "coarse", "r18", "read_A.npz"))
PIX = RA["ix"]                     # the 19,000 held + test photos, in the order of the caches (held, then test)
S, GRID = 110, 7


def window(g, cell):
    c = 224 // GRID
    w = 4 * c
    img = np.pad(np.asarray(X[int(IXT[PIX[g]])], np.float32), ((w, w), (w, w), (0, 0)), mode="reflect")
    rr, kk = divmod(int(cell), GRID)
    cy, cx = int((rr + .5) * c) + w, int((kk + .5) * c) + w
    return img[cy - w // 2:cy + w // 2, cx - w // 2:cx + w // 2]


def edges(a):
    gy, gx = np.gradient(a.mean(2))
    return np.hypot(gx, gy)


def tile(a, grey=False):
    if grey:
        a = np.repeat(np.clip(a / (np.quantile(a, .99) + 1e-6) * 255, 0, 255)[..., None], 3, 2)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((S, S))


def row(sheet, y, title, gs, cells):
    wins = [window(g, c) for g, c in zip(gs, cells)]
    dr = ImageDraw.Draw(sheet)
    dr.text((4, y + 2), title, fill="black")
    y += 14
    sheet.paste(tile(np.mean(wins, 0)), (4, y))
    sheet.paste(tile(np.mean([edges(w) for w in wins], 0), grey=True), (S + 8, y))
    for i, w in enumerate(wins[:8]):
        sheet.paste(tile(w), (2 * S + 20 + i * (S + 2), y))
    return y + S + 6


def main(pole):
    F = np.load(os.path.join(R, f"oc_firesplit_{pole}.npz"))
    C = np.load(os.path.join(R, f"oc_subcircuits_{pole}.npz"))
    dev, g_dev, cps = C["dev"], C["g_dev"], C["cps"]
    gix, cell, pre = F["gix"], F["cell"], F["pre"]
    for k in range(g_dev.max() + 1):
        mem = dev[g_dev == k]
        mem = mem[np.argsort(-np.abs(pre[mem]))]
        top3 = np.argsort(-cps[k])[:3]
        sheet = Image.new("RGB", (2 * S + 24 + 8 * (S + 2), 4 * (S + 22) + 10), "white")
        y = row(sheet, 4, f"n101 {pole} pole, subcircuit {k}: {len(mem)} DEV firings (colour mean, edge mean, crops)",
                gix[mem], cell[mem])
        for j in top3:
            own = np.argsort(-RA["pos0"][:, j].astype(np.float32))[:12]
            shared = len(set(own.tolist()) & set(gix.tolist()))
            y = row(sheet, y, f"counterpart A n{j} (mean match {cps[k][j]:.2f}) on its own top 12 photos; "
                              f"{shared} of them are n101 {pole} firings", own, RA["arg0"][own, j])
        f = os.path.join(R, f"sub_{pole}_k{k}.png")
        sheet.save(f)
        print("saved", f)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pos")
