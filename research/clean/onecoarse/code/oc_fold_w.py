"""Steps 2 and 3 of "Children by the parent's write" (LOG, 2026-11-23). Mac, numpy, saved weights only.

Step 2: each child's other work: its 3 largest positive and negative inputs besides channel 69
        (signed sum over 3 x 3 times the child's write norm). Predictions S2a, S2b in the LOG.
Step 3: fold rule r_fold = sum_i a_i r_i, a_i = positive part of the child score (all 512, no cut).
        Match = cosine with n101's read over 256 x 3 x 3. Chance = 200 shuffles of a over A neurons.
        F3a full read, F3b with channel 69 removed from both. Power: shuffled 99th pct < .5.
Output: results/oc_fold_w.npz and a printed report.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
P = 69


def cos(a, b):
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def main():
    d = np.load(os.path.join(RES, "oc_weights.npz"))
    c = np.load(os.path.join(RES, "oc_children_w.npz"))
    R = d["read_all_A"].astype(np.float64)           # 512 x 256 x 3 x 3
    r101 = d["read_c101"][0].astype(np.float64)
    score, wnorm, order = c["score"], c["wnorm"], c["order"]

    # Step 2
    inp = R.sum((2, 3)) * wnorm[:, None]               # 512 x 256, scale-free
    print("STEP 2: other inputs (channel: scale-free weight)")
    rows = {}
    for j in list(order[:20]) + [81, 57, 233]:
        v = inp[j].copy(); v[P] = np.nan
        o = np.argsort(-np.nan_to_num(v, nan=-1e9))
        neg = np.argsort(np.nan_to_num(v, nan=1e9))
        pos3 = [(int(k), round(float(v[k]), 2)) for k in o[:3]]
        neg3 = [(int(k), round(float(v[k]), 2)) for k in neg[:3]]
        rows[int(j)] = (pos3, neg3)
        print(f"n{j:<4d} score {score[j]:5.2f}  + {pos3}  - {neg3}")
    top3 = {j: [k for k, _ in rows[j][0]] for j in rows}
    print("S2a n108 has s3:88:", 88 in top3[108], "| n81:", 88 in top3[81])
    print("S2b n305 has s3:86:", 86 in top3[305], "| n57:", 86 in top3[57], "| n233:", 86 in top3[233])

    # Step 3
    a = np.clip(score, 0, None)
    def fold(w):
        return np.tensordot(w, R, axes=(0, 0))
    def no69(x):
        y = x.copy(); y[P] = 0; return y
    real = fold(a)
    f3a, f3b = cos(real, r101), cos(no69(real), no69(r101))
    rng = np.random.default_rng(0)
    sa, sb = [], []
    for _ in range(200):
        s = fold(rng.permutation(a))
        sa.append(cos(s, r101)); sb.append(cos(no69(s), no69(r101)))
    sa, sb = np.array(sa), np.array(sb)
    single = np.array([cos(R[j], r101) for j in range(512)])
    single_b = np.array([cos(no69(R[j]), no69(r101)) for j in range(512)])
    mean = cos(R.mean(0), r101)
    print("\nSTEP 3: fold vs n101's read")
    print(f"power: shuffled 99th pct F3a {np.quantile(sa, .99):.3f}, F3b {np.quantile(sb, .99):.3f} (bar < .5)")
    print(f"F3a real {f3a:.3f} vs shuffled median {np.median(sa):.3f} 99th {np.quantile(sa, .99):.3f}")
    print(f"F3b real {f3b:.3f} vs shuffled median {np.median(sb):.3f} 99th {np.quantile(sb, .99):.3f}")
    bo = np.argsort(-single)[:5]
    print("best single A:", [(f"n{j}", round(float(single[j]), 3)) for j in bo],
          "| without ch69:", [(f"n{j}", round(float(single_b[j]), 3)) for j in np.argsort(-single_b)[:5]])
    print(f"A mean read: {mean:.3f}")
    # fold on channel sums (where does r_fold put weight vs n101)
    fs, cs = real.sum((1, 2)), r101.sum((1, 2))
    print("fold top channels:", np.argsort(-fs)[:10].tolist(), "| n101 top channels:", np.argsort(-cs)[:10].tolist())
    print("fold bottom channels:", np.argsort(fs)[:6].tolist(), "| n101 bottom:", np.argsort(cs)[:6].tolist())
    np.savez(os.path.join(RES, "oc_fold_w.npz"), a=a, r_fold=real, f3a=f3a, f3b=f3b, shuf_a=sa, shuf_b=sb,
             single=single, single_b=single_b, mean=mean)


if __name__ == "__main__":
    main()
