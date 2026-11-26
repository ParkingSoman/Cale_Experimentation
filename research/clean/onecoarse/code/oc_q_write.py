# scale: neurons=s3:69, A:children, A:block1-weights, A:head-weights; net=A; photos=saved; where=mac
"""Step 3 of "Back to one coarse neuron" (2026-11-23): six writes into one, in A (weights only).

Space: A's stream after block 0. Readers there: A's block-1 neurons (their reads) and the head (Wfc through block 1's
skip and the pool). Child i's message = its write (bn2-scaled conv2 column, summed over 3 x 3) times its typical output
h_i (mean relu(peak) on parent-strong photos, saved read_A). Reader j's response = (bn1-scaled block-1 read of j,
summed over 3 x 3) . message, times j's write norm (scale-free). Head response = Wfc . message (centred over classes).
M = readers x children. Rank-1 share = s1^2 / sum s^2. Fold share = share of the columns' energy along the fold's
message (the sum of the columns). Control: the same with synthetic random write vectors of the same norms (200 draws).
Readers that tell the children apart = rows with the largest energy outside the fold's message (listed by number only).
Inputs: r18/w_A.npz, r18/read_A.npz, r18/read_s3.npz, results/oc_children_w.npz. Output: printed, results/oc_q_write.npz.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
R18 = os.path.join(HERE, "..", "..", "neuron", "results", "coarse", "r18")
P = 69


def bn_scale(bn):
    g, _, _, v = bn
    return g / np.sqrt(v + 1e-5)


def shares(M):
    s = np.linalg.svd(M, compute_uv=False)
    r1 = s[0] ** 2 / (s ** 2).sum()
    f = M.sum(1)
    f = f / np.linalg.norm(f)
    fold = ((f @ M) ** 2).sum() / (M ** 2).sum()
    return r1, fold


def main():
    w = np.load(os.path.join(R18, "w_A.npz"))
    cw = np.load(os.path.join(RES, "oc_children_w.npz"))
    rA = np.load(os.path.join(R18, "read_A.npz"))
    pos69 = np.load(os.path.join(R18, "read_s3.npz"))["pos"][:, P].astype(np.float64)
    strong = pos69 >= 2.92
    W2 = w["W2_0"] * bn_scale(w["bn2_0"])[:, None, None, None]      # stream 512 x hidden 512 x 3 x 3
    V = W2.sum((2, 3))                                                # stream x hidden
    R1 = (w["W1_1"] * bn_scale(w["bn1_1"])[:, None, None, None]).sum((2, 3))   # block-1 hidden x stream
    W2b = w["W2_1"] * bn_scale(w["bn2_1"])[:, None, None, None]
    wn1 = np.sqrt((W2b ** 2).sum((0, 2, 3)))
    Wfc = w["Wfc"] - w["Wfc"].mean(0, keepdims=True)                 # centred over classes
    rng = np.random.default_rng(0)
    res = {}
    for k in (6, 11):
        kids = cw["order"][:k]
        h = np.maximum(rA["pos0"][strong][:, kids].astype(np.float64), 0).mean(0)
        msg = V[:, kids] * h[None, :]                                 # stream x k
        cos = msg.T @ msg / np.outer(np.linalg.norm(msg, axis=0), np.linalg.norm(msg, axis=0))
        Mb = (R1 @ msg) * wn1[:, None]
        Mh = Wfc @ msg
        print(f"k={k} children {list(map(int, kids))}; typical outputs h {np.round(h, 2).tolist()}")
        print(f"   stream: max |cos| between messages {np.abs(cos - np.eye(k)).max():.3f}")
        for name, M, R in (("block-1 readers", Mb, lambda m: (R1 @ m) * wn1[:, None]), ("head", Mh, lambda m: Wfc @ m)):
            r1, fold = shares(M)
            cm = M.T @ M / np.outer(np.linalg.norm(M, axis=0), np.linalg.norm(M, axis=0))
            syn = []
            for _ in range(200):
                g = rng.standard_normal(msg.shape)
                g *= np.linalg.norm(msg, axis=0) / np.linalg.norm(g, axis=0)
                syn.append(shares(R(g)))
            syn = np.array(syn)
            f = M.sum(1) / np.linalg.norm(M.sum(1))
            resid = M - np.outer(f, f @ M)
            rowout = (resid ** 2).sum(1)
            top = np.argsort(-rowout)[:10]
            tot = (M ** 2).sum(1)
            print(f"   {name}: rank-1 share {r1:.3f} (synthetic median {np.median(syn[:, 0]):.3f}, 99th "
                  f"{np.percentile(syn[:, 0], 99):.3f}); fold share {fold:.3f} (synthetic median "
                  f"{np.median(syn[:, 1]):.3f}, 99th {np.percentile(syn[:, 1], 99):.3f}); mean off-diag cos of "
                  f"messages as read {cm[~np.eye(k, dtype=bool)].mean():+.3f}")
            if name == "block-1 readers":
                print("      readers that tell children apart (most energy outside the fold message): " +
                      ", ".join(f"b1:n{j} ({rowout[j] / tot[j]:.2f})" for j in top))
                print(f"      share of all reader energy outside the fold message: {rowout.sum() / tot.sum():.3f}")
            res[f"{name}_{k}"] = (r1, fold, syn)
    np.savez(os.path.join(RES, "oc_q_write.npz"), **{k.replace(" ", "_").replace("-", ""): np.array(v[:2])
                                                       for k, v in res.items()})


if __name__ == "__main__":
    main()
