"""What do the 4a readers want from the curve family? Stack the 4a input columns on the ten curves (368 x 10),
weight each reader row by the typical size of its input to make rows comparable, SVD. Also: the same for 10 random
other 5x5 neurons (null)."""
import numpy as np, torch
from inception import *
from family import IN4A
m = load_model(); S1 = np.load("cache/s1_fine.npz"); fam = list(S1["fam"])
np.set_printoptions(precision=2, suppress=True, linewidth=160)
def readers(idx):
    return np.concatenate([getattr(m, n).weight.detach().cpu()[:, idx, 0, 0].numpy() for n in IN4A])
def report(idx, name):
    A = readers(idx); U, s, Vt = np.linalg.svd(A, full_matrices=False)
    print(f"{name}: singular value shares {np.round(s**2 / (s**2).sum(), 3)}")
    print(f"   top right vector (over the 10 neurons) {Vt[0]}  | cos with uniform {abs(Vt[0].sum())/np.sqrt(10):.2f}")
    return Vt
Vt = report(fam, "curves (orientation order)")
rng = np.random.default_rng(0); others = [i for i in range(320, 416) if i not in fam]
for r in range(3): report(list(rng.choice(others, 10, replace=False)), f"random 10 others #{r}")
# per-branch
for n in IN4A:
    A = getattr(m, n).weight.detach().cpu()[:, fam, 0, 0].numpy(); U, s, Vt = np.linalg.svd(A, full_matrices=False)
    print(f"{n}: shares {np.round(s[:4]**2/(s**2).sum(),3)} top vector {Vt[0]} cos uniform {abs(Vt[0].sum())/np.sqrt(10):.2f}")
