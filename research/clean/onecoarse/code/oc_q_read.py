# scale: neurons=s3:69, B:n101, A:children; net=A, B128aug_s1; photos=saved; where=mac
"""Step 1 of "Back to one coarse neuron" (2026-11-23): n101's read, channel by channel.

For each stage-3 channel c that n101 reads:
  score   = signed sum of n101's bn1-scaled kernel on c over 3 x 3, times n101's write norm (R1 score form)
  share   = c's part of n101's read energy
  sd      = |kernel sum| x std over photos of c's mean (how much c moves n101's mean input across photos)
  r69     = correlation of c's photo mean with channel 69's photo mean (19,000 held + test photos) - case (a)
  kids    = the R1 children that have c among their own top 5 other inputs, with the sign they read it
Inputs: results/oc_weights.npz, results/oc_children_w.npz, neuron/results/coarse/r18/{w_A,read_s3}.npz.
Output: printed table, results/oc_q_read.npz.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
R18 = os.path.join(HERE, "..", "..", "neuron", "results", "coarse", "r18")
P = 69
K = 11


def bn_scale(bn):
    g, _, _, v = bn
    return g / np.sqrt(v + 1e-5)


def main():
    d = np.load(os.path.join(RES, "oc_weights.npz"))
    cw = np.load(os.path.join(RES, "oc_children_w.npz"))
    w = np.load(os.path.join(R18, "w_A.npz"))
    s3 = np.load(os.path.join(R18, "read_s3.npz"))["mean"].astype(np.float32)

    k = d["read_c101"][0]                                    # 256 x 3 x 3
    wn = np.sqrt((d["write_c101"] ** 2).sum())
    ksum = k.sum((1, 2))
    score = ksum * wn
    share = (k ** 2).sum((1, 2)) / (k ** 2).sum()
    sd = np.abs(ksum) * s3.std(0)
    z = (s3 - s3.mean(0)) / (s3.std(0) + 1e-8)
    r69 = (z * z[:, [P]]).mean(0)

    kids = cw["order"][:K]
    R = d["read_all_A"]
    W2 = w["W2_0"] * bn_scale(w["bn2_0"])[:, None, None, None]
    kwn = np.sqrt((W2 ** 2).sum((0, 2, 3)))
    kid_in = {}
    for i in kids:
        s = R[i].sum((1, 2)) * kwn[i]
        s[P] = 0
        top = list(np.argsort(-np.abs(s))[:5])
        kid_in[int(i)] = {int(c): float(s[c]) for c in top}

    order = np.argsort(-np.abs(score))
    print(f"n101: write norm {wn:.3f}, bias {float(d['off_c101'][0]):+.3f}; children {list(map(int, kids))}")
    print(f"channel 69: score {score[P]:+.3f}, share {share[P]:.3f}, sd {sd[P]:.3f} (rank by sd "
          f"{int((sd > sd[P]).sum())})")
    print("rank ch    score   share    sd    r69   children reading it (sign)")
    rows = []
    for rnk, c in enumerate(order[:16]):
        users = [f"n{i}{'+' if v > 0 else '-'}" for i, v in ((i, kid_in[i].get(int(c))) for i in kid_in) if v]
        print(f"{rnk:4d} {c:3d} {score[c]:+7.3f} {share[c]:7.3f} {sd[c]:6.3f} {r69[c]:+6.2f}   {' '.join(users)}")
        rows.append(c)
    sd_order = np.argsort(-sd)
    print("top 16 by sd (photo movement):", [(int(c), round(float(sd[c]), 3)) for c in sd_order[:16]])
    print("sd share of channel 69 among all channels (sum of sd):", round(float(sd[P] / sd.sum()), 4))
    for c in (78,):
        print(f"s3:{c}: score rank {int((np.abs(score) > abs(score[c])).sum())}, r69 {r69[c]:+.2f}, sd {sd[c]:.3f}")
    print("children's top other inputs:", kid_in)
    np.savez(os.path.join(RES, "oc_q_read.npz"), score=score, share=share, sd=sd, r69=r69, kids=kids,
             order=order)


if __name__ == "__main__":
    main()
