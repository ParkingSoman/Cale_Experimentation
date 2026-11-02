# serves: TQ4
"""Check of Lemma TQ4 (COARSE-THEORY.md B.Q): min over invertible T of tr(T P T^T) + c tr(T^-1 Q T^-T)
= 2 sqrt(c) sum sqrt(p_i^down q_i^up), attained by the reverse pairing. Random P, Q positive definite (k = 1..5),
direct minimisation over T (many restarts, BFGS on the entries of T) must not go below the formula (beyond 1e-7
relative), and the stated T must reach it (to 1e-10). Also reports the referee's counterexample to the first
('balancing') form: P = Q = diag(1, eps), c = 1."""
import json
import os

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(3135)
fails = []


def f(T, P, Q, c):
    Ti = np.linalg.inv(T)
    return np.trace(T @ P @ T.T) + c * np.trace(Ti @ Q @ Ti.T)


def formula(P, Q, c):
    p = np.sort(np.linalg.eigvalsh(P))[::-1]
    q = np.sort(np.linalg.eigvalsh(Q))
    return 2 * np.sqrt(c) * np.sum(np.sqrt(p * q))


def stated_T(P, Q, c):
    pv, U = np.linalg.eigh(P); o = np.argsort(pv)[::-1]; pv, U = pv[o], U[:, o]
    qv, W = np.linalg.eigh(Q); o = np.argsort(qv); qv, W = qv[o], W[:, o]
    t = (c * qv / pv) ** 0.25
    return W @ np.diag(t) @ U.T


worst_gap, n = 0.0, 0
for trial in range(120):
    k = int(rng.integers(1, 6))
    A = rng.normal(size=(k, k)); P = A @ A.T + 0.05 * np.eye(k)
    B = rng.normal(size=(k, k)); Q = B @ B.T + 0.05 * np.eye(k)
    c = float(np.exp(rng.normal()))
    F = formula(P, Q, c)
    Ts = stated_T(P, Q, c)
    if abs(f(Ts, P, Q, c) - F) > 1e-10 * max(1, F):
        fails.append(f"trial {trial}: stated T gives {f(Ts, P, Q, c)} != {F}")
    best = np.inf
    for _ in range(8):
        T0 = rng.normal(size=(k, k)) + 2 * np.eye(k)

        def g(v):
            T = v.reshape(k, k)
            if abs(np.linalg.det(T)) < 1e-12:
                return 1e12
            return f(T, P, Q, c)
        r = minimize(g, T0.ravel(), method="BFGS", options={"gtol": 1e-12, "maxiter": 20000})
        best = min(best, r.fun)
    if best < F * (1 - 1e-7):
        fails.append(f"trial {trial}: search {best} below formula {F}")
    worst_gap = max(worst_gap, (best - F) / F)
    n += 1
eps = 1e-3
P = Q = np.diag([1.0, eps])
balancing = 2 * np.trace(np.diag(np.sqrt(np.diag(P) * np.diag(Q))))
out = dict(trials=n, worst_relative_search_gap=worst_gap, counterexample=dict(eps=eps, reverse_pairing=formula(P, Q, 1.0),
           first_form_balancing=float(balancing)), fails=fails)
print(f"{n} trials; search never below the formula; worst relative excess of the search {worst_gap:.2e}")
print(f"counterexample eps={eps}: reverse pairing {formula(P, Q, 1.0):.4f} (= 4 sqrt(eps) = {4 * np.sqrt(eps):.4f}); first form {balancing:.4f}")
print("fails:", len(fails), fails[:3])
json.dump(out, open(os.path.join(HERE, "ts_tq4.json"), "w"), indent=1, default=float)
