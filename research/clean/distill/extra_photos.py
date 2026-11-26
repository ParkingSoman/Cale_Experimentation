"""Extra distillation photos (wave 3, plan 2026-11-26): about 142 ImageNet-TRAIN photos per class (every 9th), labels unused.

Source: Hugging Face evanarlian/imagenet_1k_resized_256 at revision 8de107e (the same revision as our val photos).
The train shards are sorted by class (about 19 classes per shard), so every shard is read. From each shard, keep the
rows whose rank within their class (in that shard) is a multiple of --every (9). Store the JPEG bytes as they are.
One shard on disk at a time: download, select, write extra_<k>.npz (off, data, lab, src_row), delete the shard.
Runs on the Mac (the Hugging Face token stays on the Mac). Output: ~/uva-gpu/outbox/extra/extra_<k>.npz.
Usage: .venv/bin/python research/clean/distill/extra_photos.py [--every 9]
"""
import argparse
import os

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO, REV = "evanarlian/imagenet_1k_resized_256", "8de107e08d1f61884d5a4ec2dea8740667264a60"
OUT = os.path.expanduser("~/uva-gpu/outbox/extra")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=9)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    files = sorted(s.rfilename for s in HfApi().dataset_info(REPO, revision=REV).siblings
                   if s.rfilename.startswith("data/train"))
    tot = np.zeros(1000, int)
    for k, fn in enumerate(files):
        dst = os.path.join(OUT, f"extra_{k:02d}.npz")
        if os.path.exists(dst):
            tot += np.bincount(np.load(dst)["lab"], minlength=1000)
            continue
        p = hf_hub_download(REPO, fn, repo_type="dataset", revision=REV, local_dir=OUT)
        t = pq.read_table(p)
        lab = np.array(t.column("label").to_pylist())
        keep = []
        for c in np.unique(lab):
            rows = np.where(lab == c)[0]
            keep += rows[::a.every].tolist()
        keep = np.sort(np.array(keep))
        img = t.column("image").to_pylist()
        blobs = [img[i]["bytes"] for i in keep]
        off = np.zeros(len(blobs) + 1, np.int64)
        off[1:] = np.cumsum([len(b) for b in blobs])
        np.savez(dst + ".tmp.npz", off=off, data=np.frombuffer(b"".join(blobs), np.uint8), lab=lab[keep], src_row=keep,
                 shard=np.array(fn))
        os.replace(dst + ".tmp.npz", dst)
        os.remove(p)
        tot += np.bincount(lab[keep], minlength=1000)
        print(f"{k + 1}/{len(files)} {fn}: {len(lab)} rows, kept {len(keep)}, {off[-1] / 1e6:.0f} MB; "
              f"total {tot.sum()}", flush=True)
    print(f"DONE {tot.sum()} photos, per class min {tot.min()} median {np.median(tot):.0f} max {tot.max()}", flush=True)


if __name__ == "__main__":
    main()
