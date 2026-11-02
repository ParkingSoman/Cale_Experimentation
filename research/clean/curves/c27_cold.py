import numpy as np, torch
from circ import *
def st(path):
    fm = b3_train.FamilyRA(m, fam, 1); fm.add_rb(m); fm.add_ra(m); fm.load_state_dict(torch.load(path, map_location="cpu")); return fm.to(DEV)
B = {"kbar": st("cache/x5f_factor_n2000.pt"), "4d": st("cache/c8_build_4d.pt"), "netmade256": st("cache/c23_build_netmade256.pt"),
     "real256": st("cache/c23_build_real256.pt"), "krylov_it2": st("cache/c24_best.pt"), "step35k": st("cache/c21_step_n35000.pt"),
     "T": b3_train.load_b3(m, "B3-k0-factorlocal")}
R = {k: per_image(fm, D.se) for k, fm in B.items()}; np.savez("cache/c27_cold.npz", **R)
for k, r in R.items(): print(f"{k:11s} {r.mean():.4f} (se {r.std() / np.sqrt(len(r)):.4f})")
for a, b in [("netmade256", "kbar"), ("netmade256", "4d"), ("real256", "kbar"), ("krylov_it2", "kbar"), ("step35k", "kbar"), ("krylov_it2", "T")]:
    d = R[a] - R[b]; print(f"  {a} minus {b}: {d.mean():+.4f} (paired se {d.std() / np.sqrt(len(d)):.4f})")
