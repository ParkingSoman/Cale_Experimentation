"""N1: which part of the trained change matters. Rules: VISION-LOG.md N1."""
from n_harness import *
out = {}
for key in FAMS:
    f = Fam(key); P, zp, zc = f.acts(f.XT, 1500); n = len(zp)
    P1 = np.c_[P, np.ones(n)]; G = P1.T @ P1; G += np.eye(len(G)) * 1e-3 * np.trace(G) / len(G); Uc = np.linalg.solve(G, P1.T @ zc)   # (C+1, k)
    up, wp = f.parent["u"].astype(np.float64), f.parent["w"].astype(np.float64); uL, wL = f.L["u"].astype(np.float64), f.L["w"].astype(np.float64)
    # read projection: dz as a function = P du + db ; project onto span of P[up, Uc] in activation space (fit coefs on positions)
    du = uL - up; dzf = P @ du + (f.L["b0"] - f.parent["b0"]); A = np.c_[zp, P1 @ Uc, np.ones(n)]; cr = np.linalg.lstsq(A, dzf, rcond=None)[0]
    u_proj = up + cr[0] * up + Uc[:-1] @ cr[1:1 + Uc.shape[1]]; b_proj = f.parent["b0"] + cr[0] * f.parent["b0"] + Uc[-1] @ cr[1:1 + Uc.shape[1]] + cr[-1]
    Wc = np.stack([w for w in f.wc], 1) * (GM[f.cb] / GM[f.pb])[:, None]      # child writes expressed in block-pb model units
    B = np.c_[wp, Wc]; cw = np.linalg.lstsq(B, wL - wp, rcond=None)[0]; w_proj = wp + B @ cw
    arms = {"learned": f.L, "read_only": {"u": uL, "b0": f.L["b0"], "w": wp}, "write_only": {"u": up, "b0": f.parent["b0"], "w": wL},
            "read_proj_only": {"u": u_proj, "b0": b_proj, "w": wp}, "write_proj_only": {"u": up, "b0": f.parent["b0"], "w": w_proj},
            "both_proj": {"u": u_proj, "b0": b_proj, "w": w_proj}, "read_proj+write_full": {"u": u_proj, "b0": b_proj, "w": wL}}
    out[key] = {a: round(f.share({k: (np.asarray(v, np.float32) if k != "b0" else float(v)) for k, v in nr.items()}), 3) for a, nr in arms.items()}
    print(key, out[key], f"{time.time() - t0:.0f}s", flush=True)
json.dump(out, open(os.path.join(RES, "vision-n1-parts.json"), "w"), indent=1)
