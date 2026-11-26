# scale: neurons=s3:69, B:n101, A:children; net=A, B128aug_s1; photos=saved; where=mac
"""Control for Step 2 (2026-11-23): is n101 told apart from one more reader of the parent?

T is built from the R1 children, which were picked for their weight on channel 69, so "the parent tracks T" is partly
built in. Here each child and n101 get the same unselected numbers over all 19,000 saved photos:
  corr(own peak, pos69)              how closely the neuron follows the parent
  corr(own peak, T without self)     how closely it follows the other children's joint amount
  corr(relu own peak, ...)           the same after the ReLU (what is sent)
Inputs: r18/read_s3.npz, read_A.npz, read_B128aug_s1.npz, results/oc_children_w.npz. Output: printed table.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
R18 = os.path.join(HERE, "..", "..", "neuron", "results", "coarse", "r18")
P, K = 69, 11


def corr(a, b):
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    return float((a * b).mean())


def main():
    cw = np.load(os.path.join(RES, "oc_children_w.npz"))
    kids = cw["order"][:K]
    a = cw["score"][kids]
    pos69 = np.load(os.path.join(R18, "read_s3.npz"))["pos"][:, P].astype(np.float64)
    pA = np.load(os.path.join(R18, "read_A.npz"))["pos0"][:, kids].astype(np.float64)
    p101 = np.load(os.path.join(R18, "read_B128aug_s1.npz"))["pos0"][:, 101].astype(np.float64)
    contrib = np.maximum(pA, 0) * a[None, :]
    T = contrib.sum(1)
    print("neuron   corr(peak,pos69)  corr(peak,T-self)  corr(relu peak,pos69)  corr(relu peak,T-self)")
    rows = []
    for j, i in enumerate(kids):
        Ts = T - contrib[:, j]
        r = (corr(pA[:, j], pos69), corr(pA[:, j], Ts), corr(np.maximum(pA[:, j], 0), pos69),
             corr(np.maximum(pA[:, j], 0), Ts))
        rows.append(r)
        print(f"A n{int(i):<4d}  {r[0]:+.3f}            {r[1]:+.3f}             {r[2]:+.3f}                 {r[3]:+.3f}")
    rows = np.array(rows)
    r = (corr(p101, pos69), corr(p101, T), corr(np.maximum(p101, 0), pos69), corr(np.maximum(p101, 0), T))
    print(f"B n101   {r[0]:+.3f}            {r[1]:+.3f}             {r[2]:+.3f}                 {r[3]:+.3f}")
    print("children median ", np.round(np.median(rows, 0), 3).tolist(), " min", np.round(rows.min(0), 3).tolist(),
          " max", np.round(rows.max(0), 3).tolist())


if __name__ == "__main__":
    main()
