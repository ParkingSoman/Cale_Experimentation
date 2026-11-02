"""Position-level containment of each target's on-spots by its 9 blind candidates (3000 images, 14x14; early-block
candidates are max-pooled so only their 'on' side is fair). on = z > f x (neuron's max over the set).
C = P(cand on | targ on), lift = C / cand breadth, back = P(targ on | cand on), zt = cand mean z-score at targ on
spots, zneg = cand mean z-score at targ most-negative spots. Usage: python contain.py [f]"""
import sys, os, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); f = float(sys.argv[1]) if len(sys.argv) > 1 else .5
D = np.load(os.path.join(H, "out", "scan.npz")); C = json.load(open(os.path.join(H, "..", "..", "s8", "candidates.json")))
for key in ["14_1043", "12_575", "13_404", "12_245", "12_1342"]:
    T = D[key].astype(np.float32).ravel(); t = T > f * T.max(); tn = T < f * T.min()
    print(f"\n{key}: target breadth {t.mean():.4f} (n={t.sum()}), neg spots {tn.sum()}")
    for c, (b, n) in C[key].items():
        Z = D[f"{b}_{n}"].astype(np.float32).ravel(); p = Z > f * Z.max(); zs = (Z - Z.mean()) / Z.std()
        print(f"  {c} b{b:2d} n{n:<5d} breadth {p.mean():.4f}  C {p[t].mean():.2f}  lift {p[t].mean()/max(p.mean(),1e-9):6.1f}  back {t[p].mean():.3f}  zt {zs[t].mean():+.2f}  zneg {zs[tn].mean():+.2f}  r {np.corrcoef(T, Z)[0,1]:+.2f}")
