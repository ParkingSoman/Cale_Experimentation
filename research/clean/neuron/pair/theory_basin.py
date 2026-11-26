"""theory agent, step 3: a Base-only construction rule, tested as a prediction of Small neurons.

Rule (uses only Base, fit rows): every Base neuron o "leans on" its most co-firing Base neuron among those of HIGHER
value (value = output std x write norm). basin(p) = p + every o that leans on p.
Prediction: Small s, with counterpart p (best fit-corr Base neuron), = least-squares mix of basin(p) (weights on fit).
Score: test R^2 of s (after GELU, all positions). Controls, same number of extra terms k:
  rand    p + k random Base neurons
  cofire  p + p's k top co-firing Base neurons (ignores value)
  oracle  p + k Base neurons with the highest fit corr to s itself (uses s: an upper reference)
Also full ridge on all 4,096 Base neurons (ceiling of any linear rule).
"""
import json
import numpy as np

R = "../results/pair/"
g = np.load(R + "grams.npz")
F = {k: g[f"fit_{k}"].astype(np.float64) for k in ("Cbb", "Cbs", "vs")}
T = {k: g[f"test_{k}"].astype(np.float64) for k in ("Cbb", "Cbs", "vs")}
wn = np.linalg.norm(np.load(R + "writes_V.npz")["b0"].astype(np.float64), axis=1)
sbF = np.sqrt(np.diag(F["Cbb"])); RbbF = F["Cbb"] / sbF[:, None] / sbF[None, :]; np.fill_diagonal(RbbF, -np.inf)
RbsF = F["Cbs"] / sbF[:, None] / np.sqrt(F["vs"])[None, :]
val = sbF * wn
n = 4096
lean = np.full(n, -1)
for o in range(n):
    c = np.where(val > val[o], RbbF[o], -np.inf)
    if np.isfinite(c.max()):
        lean[o] = c.argmax()
basin = [[] for _ in range(n)]
for o in range(n):
    if lean[o] >= 0:
        basin[lean[o]].append(o)


def r2(S, s, lam=1e-3):
    S = np.array(S)
    A = F["Cbb"][np.ix_(S, S)]; A = A + lam * np.trace(A) / len(S) * np.eye(len(S))
    w = np.linalg.solve(A, F["Cbs"][S, s])
    e = T["vs"][s] - 2 * w @ T["Cbs"][S, s] + w @ T["Cbb"][np.ix_(S, S)] @ w
    return 1 - e / T["vs"][s]


rng = np.random.default_rng(0)
res = []
P = RbsF.argmax(0)
for s in range(3072):
    p = P[s]
    B = basin[p][:40] if len(basin[p]) > 40 else basin[p]
    if len(basin[p]) > 40:  # keep the 40 that co-fire most with p
        B = sorted(basin[p], key=lambda o: -RbbF[p, o])[:40]
    k = len(B)
    row = dict(s=s, p=int(p), k=k, alone=r2([p], s))
    if k:
        row["basin"] = r2([p] + B, s)
        others = np.setdiff1d(np.arange(n), [p])
        row["rand"] = r2([p] + list(rng.choice(others, k, replace=False)), s)
        row["cofire"] = r2([p] + list(np.argsort(-RbbF[p])[:k]), s)
        o = np.argsort(-RbsF[:, s]); o = o[o != p][:k]
        row["oracle"] = r2([p] + list(o), s)
    res.append(row)
have = [r for r in res if r["k"] > 0]
ks = np.array([r["k"] for r in have])
print(f"Small neurons whose counterpart has a basin: {len(have)}/3072, basin size median {np.median(ks):.0f}, mean {ks.mean():.1f}")
summ = {"n_with_basin": len(have), "k_median": float(np.median(ks))}
for key in ("alone", "basin", "rand", "cofire", "oracle"):
    v = np.array([r[key] for r in have]); summ[key] = float(np.median(v))
    print(f"  test R^2 median {key:7s} {np.median(v):.3f}  mean {v.mean():.3f}")
for a, b in (("basin", "alone"), ("basin", "rand"), ("basin", "cofire"), ("oracle", "basin")):
    v = np.array([r[a] - r[b] for r in have]); summ[f"{a}-{b}"] = [float(np.median(v)), float((v > 0).mean())]
    print(f"  {a} - {b}: median gain {np.median(v):.3f}, wins {(v > 0).mean():.3f}")
alone_all = np.array([r["alone"] for r in res]); print("all 3072, p alone median", np.median(alone_all))
json.dump(dict(summary=summ, rows=res), open(R + "theory/basin.json", "w"))
