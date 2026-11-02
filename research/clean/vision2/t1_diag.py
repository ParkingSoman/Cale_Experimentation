"""T1 diag: per-pair fold vs cut KL (1,000 sealed images), slope a, R^2, norm of the mapped write."""
import sys
sys.argv = ["t1_fold.py", "none"]
exec(open("t1_fold.py").read().split("results = {}")[0])
EV = EV[::3]; LP0 = evaluate(W0, B0)
KEY = {"1_351": (0, 245), "3_623": (2, 61), "4_499": (1, 365), "5_135": (4, 212), "8_4": (6, 205), "10_1316": (9, 686), "11_1031": (10, 1172),
       "12_136": (9, 1137), "14_1326": (13, 78), "14_33": (12, 548), "14_831": (3, 92), "15_16": (14, 1138), "16_384": (15, 1162), "16_26": (15, 2142),
       "16_2062": (13, 1136), "17_2283": (16, 2886)}
pairs = [tuple(map(int, k.split("_"))) + tuple(v) for k, v in KEY.items()]
F = fits(pairs)
for p in pairs:
    a, bb, mu, r2 = F[p]; cb, cn, pb, pn = p; wc = W0[cb][:, cn]; v = back(wc, ST[cb], ST[pb])
    kc = kl(evaluate(*apply([p], {p: (a, bb, mu)}, "cut"))).mean(); kf = kl(evaluate(*apply([p], {p: (a, bb, mu)}, "fold"))).mean()
    print(f"{cb} n{cn} <- {pb} n{pn}: a {a:+.3f} b {bb:+.3f} R2 {r2:.3f} | |w| {np.linalg.norm(wc):.3f} |mapped| {np.linalg.norm(v):.3f} | KL cut {kc:.5f} fold {kf:.5f}", flush=True)
