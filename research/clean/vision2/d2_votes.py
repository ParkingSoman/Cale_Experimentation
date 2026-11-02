"""D2 look: the trained narrow neuron's vote written in the wide net's z-scored votes (ridge, description). Where are the
big coefficients: its own co-firing parts (positive?), other neurons (negative?) -- and which: same coarse class / other
sign / other coarse class?"""
from d2_lib import *
W = net("gelu_256_s0"); Nn = net("gelu_16_s0"); Z = Xr @ W["W1"].T + W["b1"]; Hr = gl(Z); ZN = Xr @ Nn["W1"].T + Nn["b1"]
Zz = (Z - Z.mean(0)) / Z.std(0); zs = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-9); Cc = zs(Hr).T @ zs(gl(ZN)) / len(Hr)
tmpl = np.concatenate([Ac + Bf.mean(1), -(Ac + Bf.mean(1))]); lab = (W["W1"] / np.linalg.norm(W["W1"], axis=1, keepdims=True) @ (tmpl / np.linalg.norm(tmpl, axis=1, keepdims=True)).T).argmax(1)
A = np.linalg.solve(Zz.T @ Zz + 1e-2 * len(Zz) * np.eye(256), Zz.T @ (ZN - ZN.mean(0)))   # (256, 16)
for j in range(6):
    a = A[:, j]; own = lab[np.argmax(Cc[:, j])]; o = np.argsort(-np.abs(a))[:12]
    r2 = 1 - ((Zz @ a - (ZN[:, j] - ZN[:, j].mean())) ** 2).mean() / ZN[:, j].var()
    rel = lambda i: "same pattern" if lab[i] == own else ("same class, other sign" if lab[i] % 8 == own % 8 else "other class")
    print(f"narrow {j} (pattern {own}; R^2 {r2:.2f}): top |coef| ->", ", ".join(f"{a[i]:+.2f} {rel(i)}" for i in o[:8]))
    pos = a > 0; print(f"     sum of + coefs: same pattern {a[(lab == own) & pos].sum():.2f}, other {a[(lab != own) & pos].sum():.2f};  sum of - coefs: same pattern {a[(lab == own) & ~pos].sum():.2f}, same class other sign {a[(lab % 8 == own % 8) & (lab != own) & ~pos].sum():.2f}, other class {a[(lab % 8 != own % 8) & ~pos].sum():.2f}")
