"""R2c (NEURON-MATH section 8): average the two k = 1 twins after fixing the ReLU scale gauge by balance."""
import numpy as np, torch
from inception import *
from family import Data, Family
from w_fisher import load_build, real_kl

m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"])
A, B_ = load_build(m, fam, "L-k1-m379-lr3e-2"), load_build(m, fam, "L-k1-r-lr3e-2")
(KA, bA, cA), (KB, bB, cB) = A.weights(), B_.weights()
# Definition 8.2: closest-twin gauge, a* = argmin |K_A - a K_B|^2 + |B_A - B_B/a|^2 (1-D, grid then refine)
BA, BB = torch.cat(cA), torch.cat(cB)
f = lambda a: float((KA - a * KB).pow(2).sum() + (BA - BB / a).pow(2).sum())
grid = np.exp(np.linspace(-5, 5, 2001)); a = grid[int(np.argmin([f(g) for g in grid]))]
print(f"closest-twin gauge a* = {a:.4f}  (distance {f(a):.4g} vs a=1 {f(1.0):.4g})")
KB, bB, cB = KB * a, bB * a, [c / a for c in cB]
cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), 0))
print(f"(a) kernel cos {cos(KA, KB):.3f}   bias A {float(bA):.1f} B {float(bB):.1f}   write cos {cos(torch.cat(cA), torch.cat(cB)):.3f}"
      f"   |K| A {float(KA.norm()):.1f} B {float(KB.norm()):.1f}")
def build(K, b, cols):
    f = Family(m, fam, 1).to(DEV); f.set_weights(K, b, cols); return f
Km, bm, cm = (KA + KB) / 2, (bA + bB) / 2, [(x + y) / 2 for x, y in zip(cA, cB)]
fmid = build(Km, bm, cm); zero = [torch.zeros(c.shape[0], device=DEV) for c in cA]
builds = {"twin A": (A, A.weights()[2], zero), "twin B": (B_, B_.weights()[2], zero),
          "avg read + avg write": (fmid, cm, zero), "avg read + A write": (fmid, cA, zero), "avg read + B write": (fmid, cB, zero)}
K = real_kl(m, fam, D, D.dv, builds); se = lambda d: d.std() / np.sqrt(len(d))
for k, v in K.items(): print(f"  {k:22s} {v.mean():.4f}")
mean_tw = (K["twin A"] + K["twin B"]) / 2
for k in ["avg read + avg write", "avg read + A write", "avg read + B write"]:
    d1, d2 = K[k] - K["twin A"], K[k] - mean_tw
    print(f"  {k:22s} vs twin A {d1.mean():+.4f} (se {se(d1):.4f})   vs twin mean {d2.mean():+.4f} (se {se(d2):.4f})")
np.savez("cache/r2c_twins.npz", **{k.replace(' ', '_'): v for k, v in K.items()})
