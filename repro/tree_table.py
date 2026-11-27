"""Result 1 (ConvNeXt-Base tree, research/clean/NEURON-LOG.md section N5): run the rescore again and compare.

Runs the original script research/clean/neuron/tree/rescore.py, unchanged, with one difference: it writes its
rescore.json to repro/out/ instead of over the committed research/clean/neuron/results/tree/rescore.json.
Then it prints the table next to the committed numbers.

Needs: torch, torchvision, numpy; the photos from repro/fetch_imagenet.py; the ConvNeXt-Base weights (torchvision
downloads them, 350 MB). A GPU (CUDA or Apple MPS) takes minutes; a CPU takes hours.
Usage (from the repository root): python repro/tree_table.py
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TREE = os.path.join(ROOT, "research", "clean", "neuron", "tree")
SAVED = os.path.join(ROOT, "research", "clean", "neuron", "results", "tree", "rescore.json")
OUT = os.path.join(ROOT, "repro", "out")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, TREE)

import common  # noqa: E402

common.OUT = OUT          # rescore.py reads OUT from common when it is imported
import rescore  # noqa: E402

print(f"device {common.DEV}", flush=True)
t0 = time.time()
rescore.main()
minutes = (time.time() - t0) / 60

got = json.load(open(os.path.join(OUT, "rescore.json")))
exp = json.load(open(SAVED))
names = [("meanfill", "mean fill"), ("randpar", "random parents"), ("tree", "tree"), ("dense", "per-size re-solve")]
print(f"\nrescore.py took {minutes:.1f} min on {common.DEV}.")
print(f"Base top-1 on the 1,500 test photos: {100 * got['base_top1']:.1f}% (committed {100 * exp['base_top1']:.1f}%)\n")
print("Top-1 on the 1,500 test photos, this run (committed run)")
print("| Keep | " + " | ".join(n for _, n in names) + " |")
print("|---|" + "---|" * len(names))
worst_top1, worst_kl = 0.0, 0.0
for fr in (0.5, 0.4, 0.25):
    cells = []
    for k, _ in names:
        g, e = got[f"{k}_{fr}"], exp[f"{k}_{fr}"]
        worst_top1 = max(worst_top1, abs(g["top1"] - e["top1"]))
        worst_kl = max(worst_kl, abs(g["kl"] - e["kl"]))
        cells.append(f"{100 * g['top1']:.1f}% ({100 * e['top1']:.1f}%)")
    print(f"| {int(fr * 100)}% | " + " | ".join(cells) + " |")
print(f"\nLargest difference from the committed run: top-1 {100 * worst_top1:.2f} points, mean KL {worst_kl:.4f}.")
print("(One photo of 1,500 is 0.07 points. A different GPU or library version can move a few photos.)")
