"""Download the ImageNet validation photos and build the arrays the research code reads.

Source: Hugging Face dataset evanarlian/imagenet_1k_resized_256 (public, no login), pinned revision below, split "val"
(2 parquet files, 870 MB, 50,000 photos with the shorter side at 256 pixels).

Output (the research code expects these paths; they are git-ignored and never committed):
  research/clean/hf/imagenet/val_224.npy     uint8 (50000, 224, 224, 3), the center 224 x 224 crop of each photo, 7.5 GB
  research/clean/hf/imagenet/val_labels.npy  int32 (50000,)
Row k is row k of the val split: file 00000 (rows 0-24,999), then file 00001.
The crop is the same as research/clean/neuron/gap/unpack.py: left = (w - 224) // 2, top = (h - 224) // 2.

Check: the script hashes the 2,500 rows that neuron/tree/rescore.py reads and compares them with the hash of the
original arrays. A match means the photos are the same bytes as in the original run.

Usage (from the repository root): python repro/fetch_imagenet.py [--parquet DIR] [--keep-parquet] [--out DIR]
"""
import argparse
import hashlib
import io
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

REPO = "evanarlian/imagenet_1k_resized_256"
REV = "8de107e08d1f61884d5a4ec2dea8740667264a60"
FILES = ["data/val-00000-of-00002-b5248be478d25e41.parquet", "data/val-00001-of-00002-85f3d9c8fa1edb63.parquet"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(ROOT, "research", "clean")
OUT = os.path.join(CLEAN, "hf", "imagenet")
# sha256 of val_224.npy rows ix_train[cal + test] in sorted order (the rows rescore.py reads), from the original file
RESCORE_ROWS_SHA256 = "00c388228c373c2081d97ef96fc8d65bb2063ab401023e4b1edb23031a25b7d6"


def crop(b):
    img = Image.open(io.BytesIO(b)).convert("RGB")
    w, h = img.size
    l, t = (w - 224) // 2, (h - 224) // 2
    return np.asarray(img.crop((l, t, l + 224, t + 224)))


def rescore_rows():
    """The val_224 rows that neuron/tree/rescore.py reads: 1,000 calibration photos and 1,500 test photos."""
    ix = np.load(os.path.join(CLEAN, "curves", "cache", "r18", "ix_train.npy"))
    lab = np.load(os.path.join(CLEAN, "curves", "cache", "r18", "lab_train.npy"))
    pos = np.zeros(len(lab), int)
    for c in range(1000):
        w = np.where(lab == c)[0]
        pos[w] = np.arange(len(w))
    fit = pos % 2 == 0
    cal = [np.where((lab == c) & fit)[0][7] for c in range(1000)]
    te = np.load(os.path.join(CLEAN, "neuron", "results", "pair", "distill", "photos_test.npy"))
    return np.unique(ix[np.concatenate([cal, te])])


def rows_sha256(X, rows):
    h = hashlib.sha256()
    for r in rows:
        h.update(np.ascontiguousarray(X[int(r)]).tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="", help="folder that already holds data/val-*.parquet (skips the download)")
    ap.add_argument("--keep-parquet", action="store_true", help="keep the downloaded parquet files")
    ap.add_argument("--check-only", action="store_true", help="only hash an existing val_224.npy")
    ap.add_argument("--out", default=OUT, help="output folder (default: research/clean/hf/imagenet, where the code reads)")
    a = ap.parse_args()
    out = a.out
    os.makedirs(out, exist_ok=True)
    dst = os.path.join(out, "val_224.npy")
    t0 = time.time()
    if not a.check_only:
        paths = []
        for fn in FILES:
            p = os.path.join(a.parquet, fn) if a.parquet else ""
            if not (p and os.path.exists(p)):
                from huggingface_hub import hf_hub_download
                p = hf_hub_download(REPO, fn, repo_type="dataset", revision=REV, local_dir=os.path.join(out, "dl"))
            paths.append(p)
            print(f"have {fn} ({os.path.getsize(p) / 1e6:.0f} MB) [{time.time() - t0:.0f} s]", flush=True)
        X = np.lib.format.open_memmap(dst + ".tmp", "w+", np.uint8, (50000, 224, 224, 3))
        L = np.zeros(50000, np.int32)
        k = 0
        for p in paths:
            f = pq.ParquetFile(p)
            for g in range(f.num_row_groups):
                t = f.read_row_group(g, columns=["image", "label"])
                for img, lab in zip(t.column("image").to_pylist(), t.column("label").to_pylist()):
                    X[k] = crop(img["bytes"])
                    L[k] = lab
                    k += 1
            print(f"decoded {k} photos [{time.time() - t0:.0f} s]", flush=True)
        assert k == 50000, k
        X.flush()
        del X
        os.replace(dst + ".tmp", dst)
        np.save(os.path.join(out, "val_labels.npy"), L)
        if not a.keep_parquet and not a.parquet:
            for p in paths:
                os.remove(p)
    X = np.load(dst, mmap_mode="r")
    rows = rescore_rows()
    got = rows_sha256(X, rows)
    ok = got == RESCORE_ROWS_SHA256
    print(f"rescore rows: {len(rows)}, sha256 {got[:16]}..., {'MATCHES the original' if ok else 'DOES NOT MATCH'} "
          f"[{time.time() - t0:.0f} s]", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
