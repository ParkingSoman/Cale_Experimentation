# serves: T18
"""Proposition T18 (the subset fold) on the exact instance of Theorem T16 (`model/t16_instance.py`).

The chain, the law and the functions J, t11_value, floor, converse_build, build_forward, d2 are those of
`t16_instance.py`: its setup is executed verbatim (everything above its main loop), so the data are identical.
Forced read columns: c_1 = K S1 (gate 1), c_2 = K S2 (gate 2). Subsets T of {1, 2}. Metric Omega = the pooled second
moment E sum_p rho(p) a a^T.

Predictions (written before the run):
  P1  T = {1, 2}: R_T = R* and J_k(R_T) equals the value of Theorem T11 (about 1e-10 relative).
  P2  every J_k(R_T) >= the floor of Theorem T12.
  P3  the minimum over T is <= the value of Theorem T11.
  P4  the build of Theorem T16 part 4 for each R_T, run as a network that computes its own gates, has D2 = J_k(R_T)
      (about 1e-8 relative).
  Reported, not predicted: where the minimum lands against the multi-start search of t16_instance.json
  (k = 2: about 1190; k = 3: about 36.9).
Result note (first run): P3 printed false at k = 3 because the minimum (T = {1, 2}) exceeds the T11 value by 2e-12
relative, the same rounding as P1 (tolerance 1e-10 there, 1e-12 in P3). The proposition is not contradicted.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, "t16_instance.py")).read().split("\nfor k in (2, 3):")[0]
exec(compile(_src, "t16_instance.py (setup)", "exec"))

Omega = torch.einsum("npd,npe,p->de", A, A, rho) / n
A_ = H @ w1
G_t = g1t * g2t
uhat = ((G_t[..., None] * z) * rho[None, :, None]).sum(1)
Chat = cov(uhat)
cols_member = {1: S1[:, 0], 2: S2[:, 0]}
cols_input = {1: R1s, 2: R2s}


def kernel_for(T, k):
    """Kernel of the Theorem T11 build in which only the reads in T are forced (member space -> input space)."""
    Cy = P @ A_ @ Chat @ A_.T @ P
    if T:
        R0, _ = torch.linalg.qr(torch.stack([cols_member[i] for i in T], 1))
        Cf = R0.T @ Chat @ R0
        Cyf = P @ A_ @ Chat @ R0
        Ce = Cy - Cyf @ torch.linalg.solve(Cf, Cyf.T)
        e, U = torch.linalg.eigh(Ce)
        Up = U[:, e.argsort(descending=True)[:k - R0.shape[1]]]
        Lam = torch.cat([R0.T, Up.T @ (P @ A_ - Cyf @ torch.linalg.solve(Cf, R0.T))], 0)
    else:
        e, U = torch.linalg.eigh(Cy)
        Lam = U[:, e.argsort(descending=True)[:k]].T @ P @ A_
    return K @ Lam.T                                                     # d x k


def proj(Kp, c):
    return Kp @ torch.linalg.solve(Kp.T @ Omega @ Kp, Kp.T @ Omega @ c)


out = {}
for k in (2, 3):
    row = {"floor (T12)": floor(k), "T11 value": t11_value(k)}
    vals = {}
    for T in ((), (1,), (2,), (1, 2)):
        Kp = kernel_for(T, k)
        R1, R2 = proj(Kp, cols_input[1]), proj(Kp, cols_input[2])
        Jv = float(J(R1, R2, k))
        cb = converse_build(R1, R2, k)
        dB = d2(build_forward(*cb))
        vals[str(set(T)) if T else "{}"] = Jv
        row[f"T={set(T) if T else '{}'}: J"] = Jv
        row[f"T={set(T) if T else '{}'}: build D2 rel err (P4)"] = abs(dB - Jv) / Jv
        if T == (1, 2):
            row["P1 |R_T - R*| max"] = float(max((R1 - R1s).abs().max(), (R2 - R2s).abs().max()))
            row["P1 rel err J vs T11"] = abs(Jv - row["T11 value"]) / row["T11 value"]
    row["P2 min J >= floor"] = min(vals.values()) >= row["floor (T12)"] * (1 - 1e-12)
    row["P3 min J <= T11"] = min(vals.values()) <= row["T11 value"] * (1 + 1e-12)
    row["subset fold value (min over T)"] = min(vals.values())
    row["best subset"] = min(vals, key=vals.get)
    out[f"k={k}"] = row

print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "t18_instance.json"), "w"), indent=1)
