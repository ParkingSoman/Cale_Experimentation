"""N6, on the cluster: rebuild hf/imagenet/val_224.npy from the jpeg_<k>.npz chunks (only the packed rows are filled).
Run from research/clean/ (the mirror root). Usage: python neuron/gap/unpack.py <folder with jpeg_*.npz>
"""
import glob
import io
import os
import sys

import numpy as np
from PIL import Image


def crop(b):
    img = Image.open(io.BytesIO(b)).convert("RGB")
    w, h = img.size
    l, t = (w - 224) // 2, (h - 224) // 2
    return np.asarray(img.crop((l, t, l + 224, t + 224)))


def main():
    src = sys.argv[1]
    os.makedirs("hf/imagenet", exist_ok=True)
    X = np.lib.format.open_memmap("hf/imagenet/val_224.npy", "w+", np.uint8, (50000, 224, 224, 3))
    n = 0
    for f in sorted(glob.glob(os.path.join(src, "jpeg_*.npz"))):
        z = np.load(f)
        ids, off, data = z["ids"], z["off"], z["data"]
        for i, vi in enumerate(ids):
            X[vi] = crop(data[off[i]:off[i + 1]].tobytes())
        n += len(ids)
        print(f, len(ids), flush=True)
    X.flush()
    print("rows filled:", n)


if __name__ == "__main__":
    main()
