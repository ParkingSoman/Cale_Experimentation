"""Score held-out-L_ex-selected iterates of C4 (it 4) and C6 (it 6) on dev-5000, paired with S and T."""
import numpy as np, torch
SRC = open("c46_readerstep.py").read(); exec(SRC.split("X = np.load(sys.argv[1])")[0])
refs = REFS(); refs["T"] = np.load("cache/c5_gapb.npz")["T"]
for tag in ["c54_gn_it6", "c50_photo2000_it8"]:
    r = per_image(build(torch.load(f"cache/{tag}_V.pt").to(DEV))); np.save(f"cache/c56_{tag}.npy", r)
    print(f"{tag}: dev-5000 {r.mean():.4f}", flush=True); paired(tag, r, {"kbar(S)": refs["kbar"], "T": refs["T"]})
print("done")
