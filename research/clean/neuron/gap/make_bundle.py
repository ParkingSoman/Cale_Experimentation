"""N6: pack what the UVA cluster needs to run train_ref.py, into ~/uva-gpu/outbox/n6/.

  jpeg_<k>.npz   JPEG bytes (from the ImageNet val parquets) of every photo in train/held/cal/test, chunks under 85 MB
  code.tar.gz    a mirror of the needed files, at the same relative paths under research/clean/
On the cluster, unpack.py rebuilds hf/imagenet/val_224.npy (only these rows filled) by the same center crop.
Check here: the crop of the parquet JPEG equals val_224.npy for every packed photo (max difference printed).
"""
import io
import os
import tarfile

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

import common as C

CLEAN = os.path.abspath(os.path.join(C.HERE, "..", ".."))
OUTBOX = os.path.expanduser("~/uva-gpu/outbox/n6")
PARQ = ["val-00000-of-00002-b5248be478d25e41.parquet", "val-00001-of-00002-85f3d9c8fa1edb63.parquet"]
FILES = ["neuron/gap/common.py", "neuron/gap/cache3.py", "neuron/gap/train_ref.py", "neuron/gap/unpack.py",
         "neuron/pair/cache.py", "neuron/results/pair/distill/photos_test.npy",
         "curves/map_agents/resnet18/net.py", "curves/cache/r18/ix_train.npy", "curves/cache/r18/lab_train.npy"]


def crop(b):
    img = Image.open(io.BytesIO(b)).convert("RGB")
    w, h = img.size
    l, t = (w - 224) // 2, (h - 224) // 2
    return np.asarray(img.crop((l, t, l + 224, t + 224)))


def main():
    os.makedirs(OUTBOX, exist_ok=True)
    ph = C.net.Photos("read")
    sets = C.photo_sets(ph.lab)
    need = np.unique(np.concatenate([ph.ix[sets[k]] for k in ("train", "held", "cal", "test")]).astype(int))
    print("photos to pack:", len(need))
    X = np.load(os.path.join(CLEAN, "hf", "imagenet", "val_224.npy"), mmap_mode="r")
    blobs, ids, size, k, worst, base = [], [], 0, 0, 0, 0
    needset = set(need.tolist())

    def flush():
        nonlocal blobs, ids, size, k
        if not ids:
            return
        off = np.cumsum([0] + [len(b) for b in blobs])
        np.savez(os.path.join(OUTBOX, f"jpeg_{k}.npz"), ids=np.array(ids), off=off,
                 data=np.frombuffer(b"".join(blobs), np.uint8))
        print(f"jpeg_{k}.npz: {len(ids)} photos, {size / 1e6:.1f} MB", flush=True)
        blobs, ids, size, k = [], [], 0, k + 1

    for name in PARQ:
        f = pq.ParquetFile(os.path.join(CLEAN, "hf", "imagenet", "data", name))
        for g in range(f.num_row_groups):
            col = f.read_row_group(g, columns=["image"]).column("image")
            for r in range(len(col)):
                vi = base + r
                if vi in needset:
                    b = col[r].as_py()["bytes"]
                    if len(ids) % 50 == 0:
                        worst = max(worst, int(np.abs(crop(b).astype(int) - X[vi]).max()))
                    blobs.append(b)
                    ids.append(vi)
                    size += len(b)
                    if size > 80e6:
                        flush()
            base += len(col)
    flush()
    print("checked crops (every 50th): max pixel difference", worst)
    with tarfile.open(os.path.join(OUTBOX, "code.tar.gz"), "w:gz") as tf:
        for p in FILES:
            tf.add(os.path.join(CLEAN, p), arcname=p)
    print("code.tar.gz written")


if __name__ == "__main__":
    main()
