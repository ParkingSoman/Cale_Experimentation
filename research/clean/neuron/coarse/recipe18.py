"""N11 step 2 reading (numpy only, light on the Mac): which of A's block-0 neurons does a coarse B neuron's job come from?
Inputs (results/coarse/r18/): read_A.npz, read_<B>.npz (capture18), path_A.npz, path_<B>.npz (path18), worth_A.npz.

For B's block-0 neurons with the highest car score (mean class-profile z over the 11 car classes):
  feeders  = its top stage-3 channels by mean contribution on car photos where it is on (feed0_cars)
  recipe   = A block-0 neurons whose feed0_cars vector has the largest cosine with B's: fed by the same channels
  profile  = A block-0 neurons whose class profile correlates most with B's (same classes)
For each A neuron listed: top classes, car split worth (worth_A split0, x 1000) and its best and worst car classes.
Usage: python recipe18.py <B tag> [n]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "coarse", "r18")
CI = json.load(open(os.path.join(HERE, "..", "results", "read", "imagenet_class_index.json")))
NAMES = [CI[str(i)][1] for i in range(1000)]
CARS = [407, 436, 468, 511, 609, 627, 656, 661, 717, 751, 817]
SHORT = ["amb", "wagon", "cab", "conv", "jeep", "limo", "minivan", "ModelT", "pickup", "racer", "sports"]


def prof(d, key):
    v = d[key].astype(np.float32)
    lab = d["lab"]
    cm = np.zeros((1000, v.shape[1]), np.float32)
    np.add.at(cm, lab, v)
    cm /= np.bincount(lab, minlength=1000)[:, None]
    return (cm - cm.mean(0)) / cm.std(0).clip(1e-6)


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-9)


def top(p, j, n=7):
    return ", ".join(NAMES[c] for c in np.argsort(-p[:, j])[:n])


def main():
    tag = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    A, B = np.load(os.path.join(OUT, "read_A.npz")), np.load(os.path.join(OUT, f"read_{tag}.npz"))
    pA, pB = prof(A, "pos0"), prof(B, "pos0")
    fA, fB = np.load(os.path.join(OUT, "path_A.npz"))["feed0_cars"], np.load(os.path.join(OUT, f"path_{tag}.npz"))["feed0_cars"]
    wA = np.load(os.path.join(OUT, "worth_A.npz"))
    split, perclass = wA["split0"] * 1000, wA["perclass0"] * 1000
    rec = unit(fB) @ unit(fA).T                       # k_B x 512
    pa = (pA - pA.mean(0)) / pA.std(0)
    pb = (pB - pB.mean(0)) / pB.std(0)
    pc = pb.T @ pa / 1000                             # k_B x 512
    car = pB[CARS].mean(0)

    def a_line(i, how, val):
        best, worst = np.argsort(-perclass[i])[:2], np.argsort(perclass[i])[:1]
        return (f"   {how} {val:.2f}  A n{i:3d}  split worth {split[i]:+.1f}  "
                f"(helps {SHORT[best[0]]} {perclass[i, best[0]]:+.0f}, {SHORT[best[1]]} {perclass[i, best[1]]:+.0f}; "
                f"hurts {SHORT[worst[0]]} {perclass[i, worst[0]]:+.0f}) | {top(pA, i, 6)}")

    for j in np.argsort(-car)[:n]:
        fe = np.argsort(-fB[j])[:6]
        print(f"\n== {tag} block 0 n{j} (car score {car[j]:.2f}) | {top(pB, j)}")
        print(f"   feeders (stage-3 channel: contribution on car photos): " + ", ".join(f"s3:{i} {fB[j, i]:+.2f}" for i in fe))
        print(f"   share of its positive car input from these 6 channels: {fB[j, fe].sum() / fB[j][fB[j] > 0].sum():.2f}")
        for i in np.argsort(-rec[j])[:4]:
            print(a_line(i, "recipe cos", rec[j, i]))
        for i in np.argsort(-pc[j])[:2]:
            print(a_line(i, "profile r ", pc[j, i]))


if __name__ == "__main__":
    main()
