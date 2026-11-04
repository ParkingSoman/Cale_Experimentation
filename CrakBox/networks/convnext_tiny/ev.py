"""Evidence tools for the ConvNeXt control, the same views as the ResNet tools. Reading photos only. Usage: ev.py deep c17:123 | edit | probe NODE IMAGE... | inputs | readers | classes WORD"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import net as cn  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "runs", "convnext_tiny", "views")
torch.set_num_threads(2)
W = cn.load_weights()
P = cn.Photos("read")
LAB = np.load(os.path.join(cn.STORE, "read_labels.npy"))


def names():
    import torchvision
    return torchvision.models.ResNet18_Weights.IMAGENET1K_V1.meta["categories"]


def arr(k):
    return np.load(os.path.join(cn.STORE, f"read_{k}.npy"), mmap_mode="r")


def field(raw, j):
    with torch.no_grad():
        _, a = cn.block_parts(W, cn.upto_last(W, cn.Photos.tensor(raw)), 3, cn.LAST)
    return a[..., j]


def deep(node):
    j = int(node.split(":")[1])
    cats = names()
    v = np.asarray(arr("x")[:, j], dtype=np.float64)
    n = np.asarray(arr("n")[:, j], dtype=np.float64)
    big, low = v.max(), n.min()
    rows = []
    for share in (1.0, .75, .5, .25, .1):
        o = np.argsort(np.abs(v - share * big))[:8]
        rows.append((f"{int(share * 100)}% of the largest firing ({big:.1f})", o, v[o]))
    o = np.argsort(n)[:8]
    rows.append((f"negative pole, strongest ({low:.1f})", o, n[o]))
    o = np.argsort(np.abs(n - .5 * low))[:8]
    rows.append(("negative pole, half strength", o, n[o]))
    S = 112
    sheet = Image.new("RGB", (8 * S, (S + 28) * len(rows) + 16), "white")
    d = ImageDraw.Draw(sheet)
    d.text((2, 2), f"{node}. On for {(v > .01 * big).mean():.2f} of the photos.", fill="black")
    y = 16
    for title, idx, vals in rows:
        d.text((2, y), title, fill="black")
        y += 12
        raw = P.raw(idx)
        for k, i in enumerate(idx):
            sheet.paste(Image.fromarray(raw[k]).resize((S, S)), (k * S, y))
            d.text((k * S + 1, y + S), f"{vals[k]:.1f} {cats[int(LAB[i])][:14]}", fill="gray")
        y += S + 16
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, node.replace(":", "_") + "_deep.png")
    sheet.save(path)
    print(path)


def edit(node):
    j = int(node.split(":")[1])
    v = np.asarray(arr("x")[:, j], dtype=np.float64)
    raw = P.raw(np.argsort(-v)[:16])
    edits = dict(flip=lambda i: i[:, ::-1], rot90=lambda i: np.rot90(i), rot180=lambda i: np.rot90(i, 2), gray=lambda i: np.repeat(i.mean(2, keepdims=True).astype(np.uint8), 3, 2),
                 blur1=lambda i: np.asarray(Image.fromarray(i).filter(ImageFilter.GaussianBlur(1))), blur3=lambda i: np.asarray(Image.fromarray(i).filter(ImageFilter.GaussianBlur(3))), invert=lambda i: 255 - i)

    def peak(b):
        return field(b, j).flatten(1).max(1).values.numpy()
    base = peak(raw)
    print(f"{node}: top 16 photos, median largest value {np.median(base):.2f}. Share kept, median:")
    print("  ".join(f"{k} {np.median(peak(np.stack([f(i).copy() for i in raw])) / np.where(base == 0, 1, base)):.2f}" for k, f in edits.items()))


def probe(node, files):
    j = int(node.split(":")[1])
    big = float(np.asarray(arr("x")[:, j]).max())
    for f in files:
        val = float(field(np.asarray(Image.open(f).convert("RGB").resize((224, 224)))[None], j).flatten(1).max())
        print(f"{f}: largest value {val:.2f}, {val / big:.2f} of the largest firing")


def adapter():
    sys.path.insert(0, ROOT)
    from networks.convnext_tiny.adapter import ConvNeXtTiny
    return ConvNeXtTiny(threads=2)


def show_inputs(node):
    net = adapter()
    v, p = net.peaks(node, "pos", "read")
    top = np.argsort(-v)[:32]
    D, nm, _, true = net.inputs(node, "read", top, p[top])
    me = D.mean(0)
    print(f"{node} on its 32 strongest photos, value {true.mean():.2f}. Largest stream directions:", [(nm[i], round(float(me[i]), 2)) for i in np.argsort(-np.abs(me))[:10]])
    print("A stream direction has no reading of its own here.")


def show_readers(node):
    cats = names()
    rows = adapter().readers(node, "read", 16)
    print("classes it pushes up:", [(cats[int(n.split(':')[1])], round(s, 3)) for n, s in rows if s > 0][:8])
    print("classes it pushes down:", [(cats[int(n.split(':')[1])], round(s, 3)) for n, s in rows if s < 0][:8])


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    if cmd == "deep":
        deep(rest[0])
    elif cmd == "edit":
        edit(rest[0])
    elif cmd == "probe":
        probe(rest[0], rest[1:])
    elif cmd == "inputs":
        show_inputs(rest[0])
    elif cmd == "readers":
        show_readers(rest[0])
    elif cmd == "classes":
        print([(i, c) for i, c in enumerate(names()) if rest[0].lower() in c.lower()])
