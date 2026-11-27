# Rerun the results

Two results can be checked by anyone. Each step below says what to type and what you will see.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ParkingSoman/Cale_Experimentation/blob/phase5-cross-platform/repro/Cale_rerun.ipynb)

The Colab notebook `Cale_rerun.ipynb` does all the steps. Pick a GPU runtime, then use Runtime → Run all.

## On your own computer

Run these from the repository root.

| Step | Type | What happens | Time |
|---|---|---|---|
| 1 | `pip install -r repro/requirements.txt` | Installs the versions used for the checked reruns | a few minutes |
| 2 | `python repro/ladder_tables.py` | Prints the Result 2 tables from the saved per-photo results. No photos and no GPU needed | seconds |
| 3 | `python repro/fetch_imagenet.py` | Downloads the ImageNet validation photos (870 MB) and builds the arrays the code reads (7.5 GB on disk). Ends with `MATCHES the original` | about 1 minute plus the download |
| 4 | `python repro/tree_table.py` | Runs the Result 1 script again and prints its table next to the committed numbers | see the table below |

## Result 1: ConvNeXt-Base tree (`research/clean/NEURON-LOG.md`, section N5)

One set of base weights plus a small config gives ConvNeXt-Base stages 3 and 4 at any size, with no training.
The script is `research/clean/neuron/tree/rescore.py`. It is not changed for the rerun.
`tree_table.py` only makes it write to `repro/out/rescore.json`, so the committed `research/clean/neuron/results/tree/rescore.json` stays as the reference.

Committed numbers (top-1 on 1,500 test photos; the full model gets 80.4%):

| Keep | mean fill | random parents | tree | per-size re-solve |
|---|---|---|---|---|
| 50% | 63.2% | 63.3% | 65.5% | 66.5% |
| 40% | 52.7% | 52.9% | 57.4% | 59.6% |
| 25% | 26.3% | 26.5% | 36.3% | 40.3% |

Measured reruns:

| Date | Computer | `rescore.py` time | Result |
|---|---|---|---|
| 2026-11-27 | Mac, Apple M5 GPU (MPS), macOS 26.6, torch 2.14.0 | 6.8 min | All 12 cells and the full model match. Mean KL matches to 4 decimals |
| 2026-11-27 | Linux, NVIDIA A16 (CUDA 13.0), torch 2.14.0+cu130, fresh clone from GitHub | 7.5 min | 11 of 12 cells match. Keep 25%, mean fill: 26.5% against 26.3% (2 photos of 1,500). Largest mean KL difference 0.0006 |

The original run was on a Mac GPU. The Linux run, from clone to table, took 17 minutes: 6 for the install, 3 for the photos, 8 for the table.
On both computers the photo hash matched the original.

## Result 2: ResNet-18 ladder (`research/clean/onecoarse/VECMATH.md`, section 7)

A formula (C) against training (R4) at four cut sizes, and the formula followed by training (R4C).
Recovery is 1 - mean KL(build) / mean KL(cut with no repair), on all test photos.

| Cut (neurons) | C | R4 | R4C |
|---|---|---|---|
| 11 | .731 | .767 | .799 |
| 64 | .721 | .774 | .797 |
| 192 | .742 | .832 | .843 |
| 384 | .714 | .904 | .910 |

`ladder_tables.py` runs the original reader `research/clean/onecoarse/code/oc_ladder_read.py` on `research/clean/onecoarse/results/oc_ladder`.
Its output is the same as the committed `READ.md` in that folder.
Then it computes the recovery again from the per-photo arrays `kl_test_*.npy` and checks it against the table.
Training the ladder again (`oc_ladder.py`, cluster jobs 6991221, 6991623 and 6991624) is not part of this check.

## What the photo step builds

| File | Content |
|---|---|
| `research/clean/hf/imagenet/val_224.npy` | All 50,000 validation photos, center 224 x 224 crop, uint8 |
| `research/clean/hf/imagenet/val_labels.npy` | Their labels |

| Item | Value |
|---|---|
| Source | Hugging Face `evanarlian/imagenet_1k_resized_256`, split `val` (public, no login) |
| Revision | `8de107e08d1f61884d5a4ec2dea8740667264a60` |
| Row order | file `val-00000` (rows 0 to 24,999), then file `val-00001` |
| Crop | left = (width - 224) // 2, top = (height - 224) // 2, as in `research/clean/neuron/gap/unpack.py` |
| Check | sha256 of the 2,500 rows that `rescore.py` reads, against the original array |

The photos are not in the repository. These small index files are:

| File | Content |
|---|---|
| `research/clean/curves/cache/r18/ix_train.npy`, `lab_train.npy` | The 35,000 reading photos (35 per class): their rows in `val_224.npy` and their labels |
| `research/clean/neuron/results/pair/distill/photos_test.npy` | The 1,500 test photos of Result 1 (positions in the reading photos) |
