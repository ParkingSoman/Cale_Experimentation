"""N0: what did L1's training change? Rules: VISION-LOG.md N0."""
from n_harness import *
gl = lambda z: z * 0.5 * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))
for key in FAMS:
    f = Fam(key); print(key, "cache+teacher", f"{time.time() - t0:.0f}s", "share parent-as-is 0, learned %.3f" % f.share(f.L), flush=True)
    P, zp, zc = f.acts(f.XT, 1500); n = len(zp); tr = np.arange(n) < n * 2 // 3; te = ~tr
    P1 = np.c_[P, np.ones(n)]; G = P1[tr].T @ P1[tr]; G += np.eye(len(G)) * 1e-3 * np.trace(G) / len(G); Uc = np.linalg.solve(G, P1[tr].T @ zc[tr]); zch = P1 @ Uc
    r2c = 1 - ((zc - zch)[te] ** 2).sum(0) / ((zc[te] - zc[te].mean(0)) ** 2).sum(0)
    zL = P @ f.L["u"] + f.L["b0"]; dz = zL - zp
    def r2(F, t):
        A = np.c_[F, np.ones(n)]; c = np.linalg.lstsq(A[tr], t[tr], rcond=None)[0]; return float(1 - ((t - A @ c)[te] ** 2).sum() / ((t[te] - t[te].mean()) ** 2).sum()), c
    r_p, c_p = r2(zp[:, None], dz); r_pc, c_pc = r2(np.c_[zp, zch], dz); r_tc, c_tc = r2(np.c_[zp, zc], dz)
    # where is dz big? by parent/child state
    hp, hc = gl(zp), gl(zc).sum(1); onp = hp > np.quantile(hp, .9); onc = hc > np.quantile(hc, .9)
    cells = {"p_on_c_on": dz[onp & onc].mean(), "p_on_c_off": dz[onp & ~onc].mean(), "p_off_c_on": dz[~onp & onc].mean(), "p_off_c_off": dz[~onp & ~onc].mean()}
    # write change in model units: dw = learned w - parent w ; regress on [w_p, w_c...] (gamma of block pb vs cb differ: compare in stream units)
    wp = f.parent["w"] * GM[f.pb]; wl = f.L["w"] * GM[f.pb]; dw = wl - wp; Wc = np.stack([w * GM[f.cb] for w in f.wc], 1)
    B = np.c_[wp, Wc]; cw = np.linalg.lstsq(B, dw, rcond=None)[0]; rw = 1 - ((dw - B @ cw) ** 2).sum() / (dw ** 2).sum()
    print(f"  child votes from parent's input: R2 {np.round(r2c, 2)} | read change dz: R2 on z_p {r_p:.2f}, + child votes(hat) {r_pc:.2f}, + true child z {r_tc:.2f}; coefs(z_p, zc_hat..) {np.round(c_pc[:-1], 3)}",
          f"| dz by state {({k: round(float(v), 2) for k, v in cells.items()})} | write change in span(w_p, w_c): R2 {rw:.2f}, coefs {np.round(cw, 3)} | |dw|/|wp| {np.linalg.norm(dw) / np.linalg.norm(wp):.2f} | db {f.L['b0'] - f.parent['b0']:.3f}", flush=True)
