"""Broden reader (broden1_224, the Network Dissection data set), as data only.

Format (from NetDissect-Lite loader/data_loader.py):
  index.csv  one row per photo: image, split, ih, iw, sh, sw, color, object, part, material, scene, texture.
             A category field holds ';'-separated mask PNG names, or plain label numbers (the label covers the whole
             photo). Scene and texture are photo-level numbers.
  images/    photos (JPEG) and masks (PNG). Mask label = R + 256 * G.
  label.csv  number, name, category ("object(123);part(4)"), frequency, coverage, syns. Number 0 = no label.

Cell labels on a G x G grid (G = 14 for s3, 7 for stage 4): for every label in a photo, its coverage of each cell
(share of the cell's mask pixels) and whether it covers the cell's centre pixel. Photo-level labels cover every cell.
"""
import csv
import os
import re

import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "broden", "broden1_224")
CATS = ("color", "object", "part", "material", "scene", "texture")


def load_index(root=ROOT):
    rows = []
    with open(os.path.join(root, "index.csv")) as f:
        for r in csv.DictReader(f):
            out = {"image": r["image"], "split": r["split"]}
            for k in ("ih", "iw", "sh", "sw"):
                out[k] = int(r[k])
            for c in CATS:
                items = [s for s in r.get(c, "").split(";") if s]
                out[c] = [int(s) if re.match(r"^\d+$", s) else s for s in items]
            out["source"] = r["image"].split("/")[0]
            rows.append(out)
    return rows


def load_labels(root=ROOT):
    """number -> (name, primary category, {category: count})."""
    lab = {}
    with open(os.path.join(root, "label.csv")) as f:
        for r in csv.DictReader(f):
            cats = {}
            for part in r["category"].split(";"):
                m = re.match(r"^(\w+)\((\d+)\)$", part.strip())
                if m:
                    cats[m.group(1)] = int(m.group(2))
            prim = max(cats, key=cats.get) if cats else ""
            lab[int(r["number"])] = (r["name"], prim, cats)
    return lab


def mask(root, name):
    a = np.asarray(Image.open(os.path.join(root, "images", name)).convert("RGB")).astype(np.int32)
    return a[:, :, 0] + 256 * a[:, :, 1]


def cell_labels(row, grids=(14, 7), root=ROOT):
    """For each grid G: (cell, label, cov, centre, cat) int arrays, one entry per (cell, label) with cov > 0 or centre.
    cat = index into CATS of the category field the label came from."""
    sh, sw = row["sh"], row["sw"]
    out = {G: ([], [], [], [], []) for G in grids}

    def add(G, idx, lab, cov, cen, ci):
        for lst, v in zip(out[G], (idx, np.full(len(idx), lab), cov, cen, np.full(len(idx), ci))):
            lst.append(v)

    for ci, c in enumerate(CATS):
        for item in row[c]:
            if isinstance(item, int):
                if item:
                    for G in grids:
                        n = G * G
                        add(G, np.arange(n), item, np.ones(n), np.ones(n, bool), ci)
                continue
            m = mask(root, item)
            assert m.shape == (sh, sw), (m.shape, sh, sw)
            for G in grids:
                assert sh % G == 0 and sw % G == 0, (sh, sw, G)
                ph, pw = sh // G, sw // G
                blk = m.reshape(G, ph, G, pw).transpose(0, 2, 1, 3).reshape(G * G, ph * pw)
                centre = m[ph // 2::ph, pw // 2::pw].reshape(-1)
                for lab in np.unique(blk):
                    if lab == 0:
                        continue
                    cov = (blk == lab).mean(1)
                    cen = centre == lab
                    idx = np.nonzero((cov > 0) | cen)[0]
                    add(G, idx, int(lab), cov[idx], cen[idx], ci)
    res = {}
    for G in grids:
        parts = out[G]
        if parts[0]:
            res[G] = tuple(np.concatenate(p) for p in parts)
        else:
            res[G] = tuple(np.zeros(0, dt) for dt in (int, int, float, bool, int))
    return res

