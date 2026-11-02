"""N8: compounding. All 12 families collapsed at once. Arms: cut, CF-indep, TR-indep, then (argv) seq-CF or seq-TR."""
from n8_lib import *
MODE = sys.argv[1]   # "indep" | "seqcf" | "seqtr"
F = sorted([Fam(os.path.basename(p)[:-4]) for p in glob.glob(os.path.join(CD, "*.npz")) if not os.path.basename(p).startswith("gn_")], key=lambda f: f.p)
parents = {f.p for f in F}; kids = {c for f in F for c in f.cs}; print("overlap parent/child:", parents & kids, "families", len(F), flush=True)
E = np.unique(np.concatenate([f.ev for f in F])); restore(); LI = logits(E); mx.eval(LI); print("eval images", len(E), f"{time.time() - t0:.0f}s", flush=True)
def cuts():
    restore()
    for f in F:
        for c, mu in zip(f.cs, f.mu): cut(c[0], c[1], mu)
def score(tag):
    k = klv(logits(E), LI); out = {"mean_kl": float(k.mean())}
    for f in F: out[f.key] = float(k[np.searchsorted(E, f.ev)].mean())
    return out
R = {}
if MODE == "indep":
    restore(); R["intact"] = score("intact"); cuts(); R["cut"] = score("cut")
    for tag, src in (("CF_indep", "gn"), ("TR_indep", "L")):
        cuts()
        for f in F:
            if src == "gn": z = np.load(os.path.join(CD, f"gn_{f.key}.npz")); set_neuron(f.pb, f.pn, z["u"], float(z["b0"]), z["w"])
            else: set_neuron(f.pb, f.pn, f.L["u"], f.L["b0"], f.L["w"])
        R[tag] = score(tag); print(tag, R[tag]["mean_kl"], f"{time.time() - t0:.0f}s", flush=True)
else:
    cuts()
    for f in F:
        f.prepare(); z = np.load(os.path.join(CD, f"gn_{f.key}.npz")) if MODE == "seqcf" else None
        init = {"u": z["u"].astype(np.float64), "b0": float(z["b0"]), "w": z["w"].astype(np.float64)} if MODE == "seqcf" else {k: (np.asarray(v, np.float64) if k != "b0" else v) for k, v in f.L.items()}
        th = f.gn(init, nit=2) if MODE == "seqcf" else f.train(init, epochs=4)
        set_neuron(f.pb, f.pn, th["u"], th["b0"], th["w"]); del f.XT; print(" solved", f.key, f"{time.time() - t0:.0f}s", flush=True)
    R[MODE] = score(MODE)
for k, v in R.items(): print(k, "mean KL %.5f" % v["mean_kl"], {kk: round(vv, 5) for kk, vv in v.items() if kk != "mean_kl"}, flush=True)
json.dump(R, open(os.path.join(RES, f"vision-n8-{MODE}{os.environ.get("TAG", "")}.json"), "w"), indent=1)
