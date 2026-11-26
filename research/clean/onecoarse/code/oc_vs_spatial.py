# scale: neurons=A:block0-all; net=A, C-ladder, WC-ladder; photos=saved; where=mac
# rung: Q=plan "Read what training adds on top of the calculation", step 1 follow-up: is the 'rest' part of WC-C a spatial reshaping (same channels, other 3x3 positions) or new channel directions; N=weights only; EST=10 s Mac; PERMIT=Shonu approved plan 2026-11-25
"""Split each part of WC - C (Dspan, Kspan, rest; as in oc_vs_writes.py) into its 3x3-uniform component (the same
512-channel vector at all 9 positions: what vecread reads) and the spatial component (sums to 0 over the 9 positions).
Random 4608-vector floor for the uniform share: 1/9 = .111."""
import os

import numpy as np

from oc_vs_writes import LAD, R18, basis, writes

VA, _ = writes(np.load(os.path.join(R18, "w_A.npz")))
print("part: share of |WC-C|^2 | uniform share within the part (floor .111)")
for n in (11, 64, 192, 384):
    d = os.path.join(LAD, f"n{n}")
    jt = np.load(os.path.join(d, "jobtest_L.npz"))
    D, K = jt["D"], jt["K"]
    QD = basis(VA[D])
    QK = basis(VA[K] - VA[K] @ QD @ QD.T)
    Dl = writes(np.load(os.path.join(d, "w_WC.npz")))[0][K] - writes(np.load(os.path.join(d, "w_C.npz")))[0][K]
    pD = Dl @ QD @ QD.T
    pK = (Dl - pD) @ QK @ QK.T
    rest = Dl - pD - pK
    tot = (Dl ** 2).sum()
    row = []
    for name, P in (("D", pD), ("K", pK), ("rest", rest), ("all", Dl)):
        X = P.reshape(len(K), 512, 9)
        uni = (X.mean(2, keepdims=True) ** 2).sum() * 9
        row.append(f"{name} {(P ** 2).sum() / tot:.2f} | {uni / (X ** 2).sum():.2f}")
    print(f"cut {n}: " + "   ".join(row))
