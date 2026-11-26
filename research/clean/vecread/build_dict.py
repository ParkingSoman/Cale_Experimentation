"""Build the concept dictionary for ResNet-18 A (torchvision IMAGENET1K_V1) from Broden.

One forward pass over Broden photos. Spaces:
  s3    stage-3 output h            256 x 14 x 14   labels on the 14 grid
  s4b0  stage-4 block-0 z = r+skip  512 x 7 x 7     labels on the 7 grid
  s4b1  stage-4 block-1 z = r+h     512 x 7 x 7     labels on the 7 grid
  s3k   stage-3 3 x 3 patches as stage 4 reads them (stride 2, padding 1): 2304 x 7 x 7, each output cell labeled
        by its centre s3 cell. Used to read 3 x 3 read kernels exactly.
Build split ("train" photos), in two halves by photo order (h0, h1: used only to merge label twins): streaming sums,
per space, label rule ("half": coverage >= .5, "centre": the label covers the cell's centre pixel) and half, per label
and source data set: S (sum of activation vectors), N (cell count), P (photo count per label), plus per-source totals
T, M and the second moment XtX (for whitening and read patterns).
Held-out split ("val" photos, up to HELD photos, seeded subset): whole maps of all three spaces (fp16) and all cell
labels, for every test in vr_check.py.
Usage: python build_dict.py [n]     (n = smoke run on n build and n held photos)
"""
import os
import sys
import time

import numpy as np
import torch
import torchvision

import broden

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SPACES = {"s3": (256, 14), "s3k": (2304, 7), "s4b0": (512, 7), "s4b1": (512, 7)}
LGRID = {"s3": 14, "s3k": 14, "s4b0": 7, "s4b1": 7}   # grid the labels come from
CENTRE7 = np.array([(2 * i) * 14 + 2 * j for i in range(7) for j in range(7)])
TO7 = -np.ones(196, int)
TO7[CENTRE7] = np.arange(49)
RULES = ("half", "centre")
HELD = 8000
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_model():
    m = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(DEV)


@torch.no_grad()
def spaces(m, x):
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    x = m.layer2(m.layer1(x))
    s3 = m.layer3(x)
    b = m.layer4[0]
    z0 = b.bn2(b.conv2(b.relu(b.bn1(b.conv1(s3))))) + b.downsample(s3)
    h0 = torch.relu(z0)
    b = m.layer4[1]
    z1 = b.bn2(b.conv2(b.relu(b.bn1(b.conv1(h0))))) + h0
    s3k = torch.nn.functional.unfold(s3, 3, padding=1, stride=2).view(len(s3), 2304, 7, 7)
    return {"s3": s3, "s3k": s3k, "s4b0": z0, "s4b1": z1}


class DS(torch.utils.data.Dataset):
    def __init__(self, rows, ids):
        self.rows, self.ids = rows, ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        from PIL import Image
        i = self.ids[k]
        r = self.rows[i]
        im = Image.open(os.path.join(broden.ROOT, "images", r["image"])).convert("RGB")
        if im.size != (224, 224):
            im = im.resize((224, 224), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
        return x, i, broden.cell_labels(r, grids=(14, 7))


def collate(b):
    return torch.stack([t[0] for t in b]), [t[1] for t in b], [t[2] for t in b]


def loader(rows, ids, bs=64):
    return torch.utils.data.DataLoader(DS(rows, ids), batch_size=bs, num_workers=int(os.environ.get("VR_WORKERS", 0)), collate_fn=collate)


def main():
    held2 = "held2" in sys.argv          # confirmation set: val photos not in the first held set, build skipped
    args = [a for a in sys.argv[1:] if a != "held2"]
    smoke = int(args[0]) if args else 0
    rows = broden.load_index()
    labels = broden.load_labels()
    L = max(labels) + 1
    sources = sorted({r["source"] for r in rows})
    src_of = np.array([sources.index(r["source"]) for r in rows])
    ns = len(sources)
    build = [i for i, r in enumerate(rows) if r["split"] == "train"]
    val = [i for i, r in enumerate(rows) if r["split"] == "val"]
    held = sorted(np.random.default_rng(0).choice(val, size=min(HELD, len(val)), replace=False).tolist())
    if held2:
        rest = sorted(set(val) - set(held))
        held = sorted(np.random.default_rng(1).choice(rest, size=min(HELD, len(rest)), replace=False).tolist())
    if smoke:
        build, held = build[:smoke], held[:smoke]
    print(f"photos: build {len(build)}, held {len(held)}; labels {L}; sources {sources}; device {DEV}", flush=True)
    m = load_model()
    tag = "_smoke" if smoke else ("_h2" if held2 else "")

    if held2:
        build = []
    # ---- build split: streaming sums
    acc = {}
    f64 = torch.float64
    for s, (C, G) in SPACES.items():
        acc[s] = {"XtX": torch.zeros((C, C), dtype=f64), "X": torch.zeros(C, dtype=f64), "n": 0}
        for h in (0, 1):
            acc[s][f"T_h{h}"] = torch.zeros((ns, C), dtype=f64)
            acc[s][f"M_h{h}"] = torch.zeros(ns, dtype=f64)
            for rule in RULES:
                acc[s][f"S_{rule}_h{h}"] = torch.zeros((L * ns, C), dtype=f64)
                acc[s][f"N_{rule}_h{h}"] = torch.zeros(L * ns, dtype=f64)
                acc[s][f"P_{rule}_h{h}"] = torch.zeros(L, dtype=f64)
    half_of = {i: k % 2 for k, i in enumerate(build)}
    t0 = time.time()
    for bi, (x, ids, labs) in enumerate(loader(rows, build)):
        A = spaces(m, ((x - MEAN) / STD).to(DEV))
        hs = np.array([half_of[i] for i in ids])
        for s, (C, G) in SPACES.items():
            X = A[s].float().cpu().permute(0, 2, 3, 1).reshape(-1, C).to(f64)
            a = acc[s]
            a["XtX"] += X.T @ X
            a["X"] += X.sum(0)
            a["n"] += len(X)
            LG = LGRID[s]
            for h in (0, 1):
                sel = np.nonzero(hs == h)[0]
                if not len(sel):
                    continue
                src_img = torch.as_tensor(src_of[np.array(ids)[sel]])
                rows_h = torch.as_tensor((sel[:, None] * G * G + np.arange(G * G)[None]).reshape(-1))
                a[f"T_h{h}"].index_add_(0, src_img.repeat_interleave(G * G), X[rows_h])
                a[f"M_h{h}"].index_add_(0, src_img, torch.full((len(sel),), float(G * G), dtype=f64))
                for rule in RULES:
                    cells, keys, plabs = [], [], []
                    for j in sel:
                        i = ids[j]
                        cell, lb, cov, cen, _ = labs[j][LG]
                        keep = (cov >= 0.5) if rule == "half" else cen
                        cell, lb = cell[keep], lb[keep]
                        if s == "s3k":
                            c7 = TO7[cell]
                            ok = c7 >= 0
                            cell, lb = c7[ok], lb[ok]
                        # one count per (cell, label): a label can come from two category fields
                        key = np.unique(lb * (G * G) + cell)
                        cells.append(j * G * G + key % (G * G))
                        keys.append((key // (G * G)) * ns + src_of[i])
                        plabs.append(np.unique(key // (G * G)))
                    cells = torch.as_tensor(np.concatenate(cells))
                    keys = torch.as_tensor(np.concatenate(keys))
                    a[f"S_{rule}_h{h}"].index_add_(0, keys, X[cells])
                    a[f"N_{rule}_h{h}"].index_add_(0, keys, torch.ones(len(keys), dtype=f64))
                    pl = torch.as_tensor(np.concatenate(plabs))
                    a[f"P_{rule}_h{h}"].index_add_(0, pl, torch.ones(len(pl), dtype=f64))
        if bi % 20 == 0:
            print(f"  build batch {bi}, {(bi + 1) * 64} photos, {time.time() - t0:.0f}s", flush=True)
    out = {"sources": np.array(sources), "label_ids": np.arange(L), "n_build": len(build)}
    for s, a in acc.items():
        for k, v in a.items():
            out[f"{s}_{k}"] = v.numpy() if torch.is_tensor(v) else v
    if not held2:
        np.savez(os.path.join(RES, f"dict_build{tag}.npz"), **out)
    print(f"build done {time.time() - t0:.0f}s", flush=True)

    # ---- held-out split: whole maps + labels
    mm = {s: np.lib.format.open_memmap(os.path.join(RES, f"held_{s}{tag}.npy"), mode="w+", dtype=np.float16,
                                       shape=(len(held), C, G, G)) for s, (C, G) in SPACES.items() if s != "s3k"}
    lab_out = {G: [[] for _ in range(6)] for G in (14, 7)}
    k0 = 0
    for x, ids, labs in loader(rows, held):
        A = spaces(m, ((x - MEAN) / STD).to(DEV))
        for s in mm:
            mm[s][k0:k0 + len(ids)] = A[s].cpu().numpy().astype(np.float16)
        for j, lab in enumerate(labs):
            for G in (14, 7):
                cell, lb, cov, cen, cat = lab[G]
                for lst, v in zip(lab_out[G], (np.full(len(cell), k0 + j), cell, lb, cov, cen, cat)):
                    lst.append(v)
        k0 += len(ids)
    for s in mm:
        mm[s].flush()
    lab_np = {}
    for G in (14, 7):
        for name, lst in zip(("img", "cell", "label", "cov", "centre", "cat"), lab_out[G]):
            lab_np[f"g{G}_{name}"] = np.concatenate(lst)
    np.savez(os.path.join(RES, f"held_labels{tag}.npz"), rows=np.array(held), source=src_of[held],
             sources=np.array(sources), **lab_np)
    print(f"held done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
