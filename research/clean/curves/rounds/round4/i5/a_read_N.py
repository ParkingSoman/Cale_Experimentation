# serves: TQ31
"""Step 1. N itself on small graphs: the principle errors of D42 and D45 on P, plus a read of one row and its units.

Fixed before the run. Base law = P (the 2m points, D43 (i)); Lebesgue on the box [-1, 3]^(2m) by 4000 samples as the D41 reading.
PREDICTION: eps_G, eps_B empty. eps_pack = 0 because a unit has one input coordinate and one route. eps_V clearly above 0. lambda > 0 for all.
"""
import sys

import numpy as np

sys.path.insert(0, "/Users/shonusengupta/Cale_Experimentation/research/clean/curves/rounds/round4/i5")
from netlib import *

np.set_printoptions(precision=3, suppress=True, linewidth=160)
graphs = {"C5": cycle(5), "C7": cycle(7), "sub(K3)": subdivide(*complete(3)), "K4": complete(4), "C6 (bipartite)": cycle(6)}
for a, rho in ((1.0, .31), (4.0, .31)):
    for name, (n, e) in graphs.items():
        U, sig = least_uncut(n, e)
        D = float(np.exp(np.ceil(np.log(n ** 2 * 50))))
        y0, mv = tq31(n, e, a, rho, D)
        layers, Wout, c = net_dq18(y0, mv)
        X = points(n)
        p = np.ones(2 * n) / (2 * n)
        chk = np.abs(forward(layers, Wout, c, X)[2] - table(y0, mv)).max()
        r = principles(layers, Wout, c, X, p)
        per = r["per"]
        lam = [q["lam"] for q in per]
        xis = [q["xi"] for q in per]
        print(f"a={a} rho={rho} {name} n={n} U={U} E={len(e)} table_err={chk:.1e} eps_V={r['eps_V']:.3f} V_rel={r['eps_V_rel']:.3f} G={r['eps_G']} B={r['eps_B']} pack={r['eps_pack']:.3f} unread={r['eps_unread']:.3f} agree={r['eps_agree']:.3f} min_lam={min(lam):.4f} h_xi[0]={xis[0]:.2e} s_xi[0]={xis[n]:.2e}", flush=True)
# Lebesgue on a box, one graph
n, e = cycle(5)
D = float(np.exp(np.ceil(np.log(n ** 2 * 50))))
y0, mv = tq31(n, e, 1.0, .31, D)
layers, Wout, c = net_dq18(y0, mv)
rng = np.random.default_rng(0)
Xb = rng.uniform(-1, 3, size=(800, 2 * n))
r = principles(layers, Wout, c, Xb, np.ones(800) / 800, want_xi=False)
print(f"Lebesgue box C5: eps_V={r['eps_V']:.3f} pack={r['eps_pack']:.3f} unread={r['eps_unread']:.3f}")
# Read one row
n, e = cycle(5)
y0, mv = tq31(n, e, 1.0, .31, D)
s = softmax(table(y0, mv))
print("row x_0^+: class, mass, logit, movement. classes: g, b0..b4, block0..4, fill0..4")
names = ["g"] + [f"b{i}" for i in range(n)] + [f"blk{i}" for i in range(n)] + [f"fil{i}" for i in range(n)]
order = np.argsort(-s[1])
for i in order[:9]:
    print(f"  {names[i]:6s} mass {s[1, i]:.4f} logit {table(y0, mv)[1, i]:.2f} move/a {(mv[0, i] * 2):.1f}")
print("h_0 col (largest):", [(names[i], round((y0 - 2 * mv)[0, i], 2)) for i in np.argsort(-(y0 - 2 * mv)[0])[:5]])
print("s_0 col (nonzero):", [(names[i], round(mv[0, i], 2)) for i in np.where(mv[0] != 0)[0]])
print("largest mass any other row's own class gets:", s[1, 2:1 + n].max())
print("g mass per row min, max:", s[:, 0].min(), s[:, 0].max())
print("tail classes that win any row:", int((s.argmax(1) > n).sum()), "of", 2 * n, "rows")
