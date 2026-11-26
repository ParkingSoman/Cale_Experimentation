"""reader, step 5: does s drop p's object (noun) and keep its local pattern (verb)? No model forward.

For each pair, top 1% rows of s and of p (equal counts, 196k rows of fwd.npz):
  n_photos   distinct photos among the rows (more = the pattern is spread over more photos)
  n_classes  distinct ImageNet classes; H = class entropy (bits)
For A = s top and p quiet ("what s adds"):
  in_p_class  fraction of A rows whose class is among p's top-row classes; in_p_photo  same for photos
  control: the same fraction for random rows (the share of rows that sit in p's classes / photos)
Output: results/pair/reader/classes.json
"""
import json
import os

import numpy as np

from cache import OUT, net

RD = os.path.join(OUT, "reader")


def H(c):
    _, n = np.unique(c, return_counts=True)
    q = n / n.sum()
    return float(-(q * np.log2(q)).sum())


def main():
    d = np.load(os.path.join(RD, "fwd.npz"))
    lab = net.Photos("read").lab
    photos = d["photos"]
    rowcls = lab[photos][np.arange(len(photos) * 49) // 49]
    rowph = np.arange(len(photos) * 49) // 49
    SV, PV = d["SV"].astype(np.float32), d["PV"].astype(np.float32)
    N = SV.shape[0]
    K = N // 100
    out, T = [], []
    for k, (s_, p_, _, _) in enumerate(d["pairs"]):
        s, p = SV[:, k], PV[:, k]
        st, pt = np.argsort(-s)[:K], np.argsort(-p)[:K]
        quiet = p < np.percentile(p, 90)
        A = st[quiet[st]]
        pc, pp = set(rowcls[pt]), set(rowph[pt])
        r = dict(k=k, s=int(s_), p=int(p_),
                 s_photos=len(set(rowph[st])), p_photos=len(pp), s_classes=len(set(rowcls[st])), p_classes=len(pc),
                 s_H=H(rowcls[st]), p_H=H(rowcls[pt]),
                 A_in_p_class=float(np.isin(rowcls[A], list(pc)).mean()),
                 A_in_p_photo=float(np.isin(rowph[A], list(pp)).mean()),
                 ctrl_in_p_class=float(np.isin(rowcls, list(pc)).mean()),
                 ctrl_in_p_photo=float(np.isin(rowph, list(pp)).mean()))
        out.append(r)
        T.append([r["s_photos"], r["p_photos"], r["s_classes"], r["p_classes"], r["s_H"], r["p_H"],
                  r["A_in_p_class"], r["ctrl_in_p_class"], r["A_in_p_photo"], r["ctrl_in_p_photo"]])
        print(k, " ".join(f"{a}={b:.2f}" if isinstance(b, float) else f"{a}={b}" for a, b in r.items()), flush=True)
    T = np.array(T, float)
    names = ["s_photos", "p_photos", "s_classes", "p_classes", "s_H", "p_H", "A_in_p_class", "ctrl_in_p_class",
             "A_in_p_photo", "ctrl_in_p_photo"]
    summ = dict(zip(names, np.median(T, 0).round(3).tolist()))
    summ["frac_s_more_classes"] = float((T[:, 2] > T[:, 3]).mean())
    summ["frac_s_more_photos"] = float((T[:, 0] > T[:, 1]).mean())
    print(summ)
    json.dump(dict(summary=summ, pairs=out), open(os.path.join(RD, "classes.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
