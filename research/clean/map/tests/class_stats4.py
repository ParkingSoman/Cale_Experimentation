"""Class-level evidence from the 20k analysis cache: per-class mean of spatial max; on-rate from m3 matrices."""
from lib4 import *
A = acts(); X, y, ana = data(); nm = names(); Mm = np.load(os.path.join(HF, "imagenet", "m3_matrices.npz"))
for b, n in [(12, 136), (14, 1326), (14, 33), (14, 831)]:
    v = A[f"max{b}"][:, n].astype(np.float32); w = A[f"min{b}"][:, n].astype(np.float32); yy = y[ana]
    cm = np.bincount(yy, v, 1000) / np.bincount(yy, minlength=1000)
    o = np.argsort(-cm)
    print(f"\n== b{b} n{n} rate={Mm[f'{b}_rate'][n]:.4f}  img-max: median {np.median(v):.2f} p90 {np.percentile(v,90):.2f} max {v.max():.2f}; frac img max>0: {(v>0).mean():.2f}; img-min median {np.median(w):.2f}")
    print(" top classes:", ", ".join(f"{nm[c]} {cm[c]:.2f}" for c in o[:15]))
    print(" bottom classes:", ", ".join(f"{nm[c]} {cm[c]:.2f}" for c in o[-10:]))
