"""One cold test on the sealed split for the Part VI builds."""
import numpy as np, torch
from circ import *
def st(path):
    fm = b3_train.FamilyRA(m, fam, 1); fm.add_rb(m); fm.add_ra(m); fm.load_state_dict(torch.load(path, map_location="cpu")); return fm.to(DEV)
B = {"level0p": st("cache/x2_b_level0_p.pt"), "w16.8": st("cache/x5h_wmetric_n2000.pt"), "4a": st("cache/c1_build_a.pt"),
     "4b": st("cache/c8_build_4b.pt"), "4d": st("cache/c8_build_4d.pt"), "5b": st("cache/c8_build_5b.pt"),
     "kbar": st("cache/x5f_factor_n2000.pt"), "w4d_local": b3_train.load_b3(m, "B3-k0-w4d-local"), "T": b3_train.load_b3(m, "B3-k0-factorlocal")}
R = {k: per_image(fm, D.se) for k, fm in B.items()}; np.savez("cache/c15_cold.npz", **R)
for k, r in R.items(): print(f"{k:10s} {r.mean():.4f} (se {r.std() / np.sqrt(len(r)):.4f})")
for a, b in [("4d", "kbar"), ("4a", "4b"), ("4b", "4d"), ("5b", "4d"), ("4d", "level0p"), ("4d", "w16.8"), ("w4d_local", "T"), ("w4d_local", "4d")]:
    d = R[a] - R[b]; print(f"  {a} minus {b}: {d.mean():+.4f} (paired se {d.std() / np.sqrt(len(d)):.4f})")
