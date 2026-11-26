"""N11: pack what the UVA cluster needs for the ResNet-18 runs, into ~/uva-gpu/outbox/n11/. CPU only, no torch.
  jpeg18_<k>.npz  JPEG bytes of the r18 test photos (odd positions) that the N6 bundle does not hold, chunks under 80 MB
  code18.tar.gz   the N11 code and index files, at their paths under research/clean/
  resnet18-f37072fd.pth  torchvision weights (the cluster has no copy)
"""
import os
import shutil
import sys
import tarfile

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(CLEAN, "curves", "map_agents", "resnet18"))
OUTBOX = os.path.expanduser("~/uva-gpu/outbox/n11")
PARQ = ["val-00000-of-00002-b5248be478d25e41.parquet", "val-00001-of-00002-85f3d9c8fa1edb63.parquet"]
FILES = ["neuron/coarse/r18.py", "neuron/coarse/cache18.py", "neuron/coarse/train18c.py", "neuron/coarse/train18.py", "neuron/coarse/unpack18.py",
         "neuron/coarse/capture18.py", "neuron/coarse/worth18.py", "neuron/coarse/edge18.py",
         "neuron/coarse/poles18.py", "curves/map_agents/resnet18/net.py",
         "curves/cache/r18/ix_train.npy", "curves/cache/r18/lab_train.npy"]


def photo_sets(lab):
    tr, held, te = [], [], []
    for c in range(1000):
        w = np.where(lab == c)[0]
        fit, odd = w[0::2], w[1::2]
        held += fit[8:10].tolist()
        tr += [i for k, i in enumerate(fit) if k not in (8, 9)]
        te += odd.tolist()
    return dict(train=np.array(tr), held=np.array(held), test=np.array(te))


def main():
    os.makedirs(OUTBOX, exist_ok=True)
    ix = np.load(os.path.join(CLEAN, "curves/cache/r18/ix_train.npy"))
    lab = np.load(os.path.join(CLEAN, "curves/cache/r18/lab_train.npy"))
    sets = photo_sets(lab)
    # what the N6 bundle packed: every fit-half photo (even position in its class) and the 1,500 N6 test photos
    fit = np.concatenate([sets["train"], sets["held"]])
    n6_test = np.load(os.path.join(CLEAN, "neuron/results/pair/distill/photos_test.npy"))
    pos = np.zeros(len(lab), int)
    for c in range(1000):
        w = np.where(lab == c)[0]
        pos[w] = np.arange(len(w))
    have = set(ix[pos % 2 == 0].astype(int).tolist()) | set(ix[n6_test].astype(int).tolist())
    need = set(ix[sets["test"]].astype(int).tolist()) - have
    assert set(ix[sets["train"]].astype(int).tolist()) <= have and set(ix[sets["held"]].astype(int).tolist()) <= have
    print("test photos to pack:", len(need), "(already on the cluster:", len(sets["test"]) - len(need), ")", flush=True)
    blobs, ids, size, k, base = [], [], 0, 0, 0

    def flush():
        nonlocal blobs, ids, size, k
        if not ids:
            return
        off = np.cumsum([0] + [len(b) for b in blobs])
        np.savez(os.path.join(OUTBOX, f"jpeg18_{k}.npz"), ids=np.array(ids), off=off,
                 data=np.frombuffer(b"".join(blobs), np.uint8))
        print(f"jpeg18_{k}.npz: {len(ids)} photos, {size / 1e6:.1f} MB", flush=True)
        blobs, ids, size, k = [], [], 0, k + 1

    for name in PARQ:
        f = pq.ParquetFile(os.path.join(CLEAN, "hf", "imagenet", "data", name))
        for g in range(f.num_row_groups):
            col = f.read_row_group(g, columns=["image"]).column("image")
            for r in range(len(col)):
                if base + r in need:
                    b = col[r].as_py()["bytes"]
                    blobs.append(b)
                    ids.append(base + r)
                    size += len(b)
                    if size > 75e6:
                        flush()
            base += len(col)
    flush()
    with tarfile.open(os.path.join(OUTBOX, "code18.tar.gz"), "w:gz") as tf:
        for p in FILES:
            tf.add(os.path.join(CLEAN, p), arcname=p)
    shutil.copy(os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"), OUTBOX)
    print("code18.tar.gz and weights written")


if __name__ == "__main__":
    main()
