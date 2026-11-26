"""Children by the parent's write (Shonu, 2026-11-23): weights only, Mac, numpy.

The parent s3:69 is a stage-3 stream channel. Its write is one stream direction (channel 69 at each place).
A child = an A stage-4 block-0 neuron whose read puts weight on channel 69.
Score (scale-free): signed sum of the bn1-scaled read on channel 69 over the 3 x 3 window, times the neuron's write
norm (bn2-scaled conv2 column). A ReLU neuron can scale its read up and its write down with no change to the net,
so the product is the fair score. Also saved: per-offset reads (the "portions"), channel 69's share of each read's
energy and its rank among the neuron's 256 inputs, n101's same numbers, and the skip path size.
Inputs: results/oc_weights.npz (read_all_A, read_c101, write_c101), ../neuron/results/coarse/r18/w_A.npz.
Output: results/oc_children_w.npz and a printed table.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
P = 69


def bn_scale(bn):
    g, _, _, v = bn
    return g / np.sqrt(v + 1e-5)


def main():
    d = np.load(os.path.join(RES, "oc_weights.npz"))
    w = np.load(os.path.join(HERE, "..", "..", "neuron", "results", "coarse", "r18", "w_A.npz"))
    R = d["read_all_A"]                                        # 512 x 256 x 3 x 3, bn1 folded
    s1 = bn_scale(w["bn1_0"])
    assert np.abs(w["W1_0"] * s1[:, None, None, None] - R).max() < 1e-5, "bn order"
    W2 = w["W2_0"] * bn_scale(w["bn2_0"])[:, None, None, None]  # out 512 x hidden 512 x 3 x 3
    wnorm = np.sqrt((W2 ** 2).sum((0, 2, 3)))
    raw = R[:, P].sum((1, 2))
    score = raw * wnorm
    energy = (R ** 2).sum((1, 2, 3))
    share = (R[:, P] ** 2).sum((1, 2)) / energy
    insum = np.abs(R.sum((2, 3)))
    rank = (insum > np.abs(raw)[:, None]).sum(1)
    order = np.argsort(-score)

    c = d["read_c101"][0]
    wc = d["write_c101"][:, 0]
    c_score = c[P].sum() * np.sqrt((wc ** 2).sum())
    c_share = (c[P] ** 2).sum() / (c ** 2).sum()
    c_rank = int((np.abs(c.sum((1, 2))) > abs(c[P].sum())).sum())

    skip = w["Wd"][:, P, 0, 0] * bn_scale(w["bnd"])
    block = (W2.sum((2, 3)) * raw[None, :]).sum(1)

    np.savez(os.path.join(RES, "oc_children_w.npz"), score=score, raw=raw, wnorm=wnorm, share=share, rank=rank,
             order=order, portions=R[:, P], c101_score=c_score, c101_share=c_share, c101_rank=c_rank,
             skip_norm=np.linalg.norm(skip), block_norm=np.linalg.norm(block))

    print("rank  neuron  score  raw  ch69-share  ch69-rank-in-own-read  portions (3x3, row-major)")
    for k, j in enumerate(order[:20]):
        print(f"{k:3d}  n{j:<4d}  {score[j]:5.2f}  {raw[j]:5.2f}  {share[j]:.3f}  {rank[j]:3d}  "
              f"{np.round(R[j, P].ravel(), 2).tolist()}")
    print("most negative:", [(f"n{j}", round(float(score[j]), 2)) for j in order[-8:][::-1]])
    print("score quantiles .5 .9 .99 max:", np.round(np.quantile(score, [.5, .9, .99, 1]), 2).tolist())
    print(f"n101: score {c_score:.2f}, share {c_share:.3f}, rank of ch69 in its read {c_rank}")
    print(f"skip path norm {np.linalg.norm(skip):.2f}, block-0 path norm (all on) {np.linalg.norm(block):.2f}")


if __name__ == "__main__":
    main()
