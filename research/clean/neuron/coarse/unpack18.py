"""N11, on the cluster: fill the r18 test rows of hf/imagenet/val_224.npy from jpeg18_<k>.npz (same center crop as
neuron/gap/unpack.py). Run from the mirror root research/clean/. Usage: python neuron/coarse/unpack18.py <folder>"""
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
    X = np.load("hf/imagenet/val_224.npy", mmap_mode="r+")
    n = 0
    for f in sorted(glob.glob(os.path.join(sys.argv[1], "jpeg18_*.npz"))):
        z = np.load(f)
        ids, off, data = z["ids"], z["off"], z["data"]           # read each array once (np.load of an npz is lazy)
        for i, vi in enumerate(ids):
            X[vi] = crop(data[off[i]:off[i + 1]].tobytes())
        n += len(ids)
        print(f, len(ids), flush=True)
    X.flush()
    print("rows filled:", n)


if __name__ == "__main__":
    main()
