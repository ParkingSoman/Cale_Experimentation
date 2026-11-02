# serves: B4.1
"""Numeric check of Remark B4.1 (COARSE-THEORY.md B.4).
Predictions: (1) the largest first coefficient |alpha| of a non-negative p(t) = 1 + 2 Re(alpha e^{it}) + 2 Re(beta e^{2it})
is cos(pi/4) (linear program on a dense grid, alpha real without loss); (2) min over phi of max_m -cos(2 pi m/5 - phi)
is cos(pi/5); (3) no non-negative p of degree 2 gives a non-negative merge of the bump: over 20000 random non-negative
p (|q|^2 for random degree-2 polynomials q, normalised), min_m a_m < 0 every time.
"""
import numpy as np
from scipy.optimize import linprog

t = np.linspace(0, 2 * np.pi, 20001)[:-1]
A = -np.stack([2 * np.cos(t), 2 * np.cos(2 * t), -2 * np.sin(2 * t)], axis=1)   # -(p - 1) <= 1
r = linprog(c=[-1, 0, 0], A_ub=A, b_ub=np.ones_like(t), bounds=[(None, None)] * 3)
print("(1) max alpha", -r.fun, "cos(pi/4)", np.cos(np.pi / 4))
phis = np.linspace(0, 2 * np.pi, 200001)
psi = 2 * np.pi * np.arange(5) / 5
print("(2)", np.min(np.max(-np.cos(psi[None, :] - phis[:, None]), axis=1)), "cos(pi/5)", np.cos(np.pi / 5))
rng = np.random.default_rng(0)
worst = -np.inf
for _ in range(20000):
    qc = rng.normal(size=3) + 1j * rng.normal(size=3)            # q(t) = sum_{n=0..2} qc_n e^{int}; p = |q|^2 >= 0
    c0 = np.sum(np.abs(qc) ** 2)
    alpha = (qc[1] * np.conj(qc[0]) + qc[2] * np.conj(qc[1])) / c0
    a = (1 + np.cos(psi - np.angle(alpha)) / np.abs(alpha)) / 5
    worst = max(worst, a.min())
print("(3) largest min_m a_m over random non-negative p:", worst, "(negative = no non-negative merge)")
