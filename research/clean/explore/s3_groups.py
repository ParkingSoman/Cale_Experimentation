"""Move 5 of the exploration: a test that can fail, inside the unit's domain (natural images). Two groups of ImageNet
classes chosen BEFORE the run by a property of the thing pictured; the unit's per-image largest value (all 50
validation images of each class) is compared between the groups. Lab work.

usage: s3_groups.py <block> <channel>
Prediction for 4e:55 (written before the run): breeds and foxes with a pointed snout (group A) score higher than
flat-faced breeds (group B): group means A > B, and at least 8 of the 10 group-B classes lie below the median class
of group A. Expected exception: Japanese spaniel (flat face, but a white blaze between dark patches, seen in move 1).
"""
import os
import sys

import numpy as np
import torch

from lib import OUT, class_names, load_images, load_model, prep, run

blk, ch = sys.argv[1], int(sys.argv[2])
A = ["collie", "Shetland sheepdog", "Pomeranian", "Siberian husky", "Eskimo dog", "basenji", "dingo", "Chihuahua",
     "papillon", "Pembroke", "Cardigan", "German shepherd", "kelpie", "red fox", "kit fox", "Arctic fox", "grey fox",
     "Samoyed", "keeshond", "malamute"]
B = ["pug", "boxer", "bull mastiff", "French bulldog", "Pekinese", "Shih-Tzu", "Boston bull", "Brabancon griffon",
     "Japanese spaniel", "Saint Bernard"]
names = class_names()


def find(nm):
    hits = [i for i, s in enumerate(names) if s.split(",")[0].strip().lower() == nm.lower()]
    assert len(hits) >= 1, nm
    return min(hits)  # animals come first in the label order (Cardigan the dog, not the sweater)


m = load_model()
X, y = load_images()
score = {}
with torch.no_grad():
    for nm in A + B:
        ix = np.where(y == find(nm))[0]
        z = run(m, prep(X[ix]), want_pre=(blk,))["pre_" + blk][:, ch]
        score[nm] = z.reshape(len(ix), -1).max(1)[0].cpu().numpy()
mA = np.array([score[k].mean() for k in A])
mB = np.array([score[k].mean() for k in B])
med = np.median(mA)
print(f"{blk}:{ch} class means of the per-image largest pre-ReLU")
print(f"group A (pointed) mean {mA.mean():.1f}, median class {med:.1f}; group B (flat) mean {mB.mean():.1f}")
print(f"group-B classes below the median group-A class: {(mB < med).sum()} of {len(B)}")
for k in sorted(A + B, key=lambda k: -score[k].mean()):
    print(f"  {'A' if k in A else 'B'} {k:20s} {score[k].mean():8.1f}")
np.savez(os.path.join(OUT, f"{blk}_{ch}_groups.npz"), **{k: v for k, v in score.items()})
