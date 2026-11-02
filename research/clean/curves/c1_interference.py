"""C1 (weights only, no images). Theorem 2: <T_s K_i, T_t K_j> = error-weighted co-activation of member i with the
shifted member j. Prediction (written first): the shifts that maximize ||sum_i T_{s_i} K_i||^2 put the members on a
ring, each displaced toward its own centre of curvature (direction (sin t, -cos t) of PREF), radius ~1 position.
Coordinate ascent over integer shifts in [-R, R]^2, 200 random restarts; the common translation is removed."""
import numpy as np, torch
from inception import load_model
from derive import PREF
m = load_model(); fam = list(np.load("cache/s1_fine.npz")["fam"])
K = m.mixed3b_5x5_pre_relu_conv.weight.detach().cpu().double().numpy()[[n - 320 for n in fam]]   # (10,32,5,5)
n = len(K); R = 3; S = [(dy, dx) for dy in range(-R, R + 1) for dx in range(-R, R + 1)]
# cross-correlation C[i,j][d] = <K_i, T_d K_j> for displacement d in [-4,4]^2 (zero padded)
P = np.zeros((n, 32, 13, 13)); P[:, :, 4:9, 4:9] = K
def ip(i, j, d):
    dy, dx = d
    if abs(dy) > 4 or abs(dx) > 4: return 0.0
    return float((P[i] * np.roll(np.roll(P[j], dy, 1), dx, 2)).sum())
C = {}
for i in range(n):
    for j in range(n):
        for dy in range(-8, 9):
            for dx in range(-8, 9):
                C[i, j, dy, dx] = ip(i, j, (dy, dx)) if (abs(dy) <= 4 and abs(dx) <= 4) else 0.0
def energy(s):
    return sum(C[i, j, s[j][0] - s[i][0], s[j][1] - s[i][1]] for i in range(n) for j in range(n))
rng = np.random.default_rng(0); best = None
for rs in range(200):
    s = [S[k] for k in rng.integers(0, len(S), n)]
    for sweep in range(20):
        changed = False
        for i in range(n):
            vals = []
            for c in S:
                t = list(s); t[i] = c; vals.append(energy(t))
            c = S[int(np.argmax(vals))]
            if c != s[i]: s[i] = c; changed = True
        if not changed: break
    e = energy(s)
    if best is None or e > best[0]: best = (e, list(s))
e0 = energy([(0, 0)] * n); e, s = best; s = np.array(s, float); s -= s.mean(0)
exp = np.stack([np.sin(PREF), -np.cos(PREF)], 1)          # (dy, dx) toward centre of curvature (derive.py convention)
# member j is READ at q + s_j in the sum K' = sum T K; derive.py reads member at q + r*exp. Compare directions.
ang = np.arctan2(s[:, 0], s[:, 1]); angx = np.arctan2(exp[:, 0], exp[:, 1])
rad = np.linalg.norm(s, axis=1)
print("energy zero-shift %.3f  best %.3f  (ratio %.2f); sum of self-energies %.3f" % (e0, e, e / e0, sum(C[i, i, 0, 0] for i in range(n))))
print("pref deg", np.rad2deg(PREF).round(0))
print("shift (dy,dx) after centring:\n", s.round(2))
print("radius per member", rad.round(2), "mean %.2f" % rad.mean())
print("angle diff to curvature-centre direction (deg):", np.rad2deg(np.angle(np.exp(1j * (ang - angx)))).round(0))
print("circular corr cos(mean diff) = %.3f  (resultant length %.3f)" % (np.cos(np.angle(np.mean(np.exp(1j * (ang - angx))))), np.abs(np.mean(np.exp(1j * (ang - angx))))))
np.savez("cache/c1_shifts.npz", s=s, e=e, e0=e0)
