"""BEAT: the WHAT at k = 3 in the second-order child-level model (Theorem 10b/11), no images (cached Phi, Sigma).
share(Lam) = tr((Lam S Lam^T)^-1 Lam S Phi S Lam^T) / tr(Phi C): the part of the family's message 3 units carry
(beta carries mu^T Phi mu / tr(Phi C) on top). Compares: Eckart-Young ceiling, 1 + cos bumps, kept members, M's
trained Lam, and every split of the 10 members (angle order) into 3 contiguous arcs with Lam >= 0 on its arc."""
import itertools, numpy as np
from scipy.optimize import minimize
d = np.load("cache/t4.npz"); S, mu = d["Szz"], d["mu"]; Phi = np.load("cache/t2.npz")["Phi"]
C = S + np.outer(mu, mu); tot = np.trace(Phi @ C); beta = mu @ Phi @ mu / tot
PREF = np.deg2rad([10, 50, 70, 120, 170, 180, 250, 290, 320, 350]); ANG = [10, 50, 70, 120, 170, 180, 250, 290, 320, 350]
def share(L):
    G = L @ S @ L.T; return np.trace(np.linalg.solve(G, L @ S @ Phi @ S @ L.T)) / tot
ev = np.sort(np.linalg.eigvals(np.linalg.solve(S, S @ Phi @ S)).real)[::-1]
print(f"beta share {beta:.3f} | total family message = 1")
print("Eckart-Young ceiling (any Lam, any sign), k = 1..4:", np.round(np.cumsum(ev[:4]) / tot, 3))
bump = lambda phis: np.stack([1 + np.cos(PREF - np.deg2rad(p)) for p in phis])
print(f"bumps 1+cos at 10/130/250: {share(bump([10, 130, 250])):.3f}; best rotation of bumps: "
      f"{max(share(bump([a, a + 120, a + 240])) for a in range(0, 120, 5)):.3f}")
eye = np.eye(10)
print(f"kept members 330/388/385 (10/120/250): {share(eye[[0, 3, 6]]):.3f}; best kept triple: "
      f"{max(share(eye[list(t)]) for t in itertools.combinations(range(10), 3)):.3f}")
u = np.load("cache/u1_k3.npz"); LM = np.stack([u[f"L-k3-m-lr3e-2_{j}"] for j in range(3)])
print(f"M's trained Lam (regression on children): {share(LM):.3f}; clipped >= 0: {share(np.clip(LM, 0, None)):.3f}")
def best_on(blocks, starts=4, seed=0):
    rng = np.random.default_rng(seed); best = (-1, None)
    def unpack(x):
        L = np.zeros((3, 10))
        o = 0
        for j, b in enumerate(blocks): L[j, b] = x[o:o + len(b)] ** 2; o += len(b)
        return L
    n = sum(len(b) for b in blocks)
    for s in range(starts):
        x0 = np.ones(n) if s == 0 else rng.uniform(.2, 1.5, n)
        r = minimize(lambda x: -share(unpack(x) + 1e-9), x0, method="L-BFGS-B")
        if -r.fun > best[0]: best = (-r.fun, unpack(r.x))
    return best
rows = []
for cuts in itertools.combinations(range(10), 3):          # arc j = members cuts[j] .. cuts[j+1]-1 (cyclic)
    blocks = [[(cuts[j] + t) % 10 for t in range((cuts[(j + 1) % 3] - cuts[j]) % 10 or 10)] for j in range(3)]
    if sorted(sum(blocks, [])) != list(range(10)): continue
    eq = np.zeros((3, 10))
    for j, b in enumerate(blocks): eq[j, b] = 1
    rows.append((share(eq), blocks))
rows.sort(key=lambda r: -r[0])
print(f"\n{len(rows)} contiguous 3-arc partitions. Equal weights inside arcs: best {rows[0][0]:.3f}, median {rows[len(rows)//2][0]:.3f}, worst {rows[-1][0]:.3f}")
res = []
for s_eq, blocks in rows[:25]:
    s_opt, L = best_on(blocks); res.append((s_opt, s_eq, blocks, L))
res.sort(key=lambda r: -r[0])
for s_opt, s_eq, blocks, L in res[:8]:
    print(f"  opt {s_opt:.3f} (equal {s_eq:.3f}) arcs {[[ANG[i] for i in b] for b in blocks]}")
Mblocks = [[8, 9, 0, 1], [2, 3, 4, 5], [6, 7]]   # nearest-start cells of 10/120/250 (M's start)
s_opt, LMb = best_on(Mblocks); print(f"M's start cells {[[ANG[i] for i in b] for b in Mblocks]}: opt {s_opt:.3f}")
np.set_printoptions(precision=3, suppress=True, linewidth=150)
print("best partition weights:\n", res[0][3] / res[0][3].sum(1, keepdims=True))
np.savez("cache/u5_what.npz", top_L=np.stack([r[3] for r in res[:4]]), M_L=LMb, bump=bump([10, 130, 250]))
