"""D1 step 1b: scatter of the family plane (positions where parent or child is on), colored by each coarse neuron,
plus a matched check: at the same parent level (5 strata among parent-on positions), coarse mean when child on minus
when child off, in units of the coarse neuron's std. usage: python d1_scatter.py KEY [child index]"""
import os, sys
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import erf
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..", "hf", "imagenet", "d1"); FIG = os.path.join(HERE, "figs", "d")
gelu = lambda z: 0.5 * z * (1 + erf(z / np.sqrt(2)))
key = sys.argv[1]; j = int(sys.argv[2]) if len(sys.argv) > 2 else 0; d = np.load(os.path.join(D, key + ".npz"))
hp, hc = gelu(d["zp"]), gelu(d[f"zc{j}"]); onp, onc = hp > .05 * hp.max(), hc > .05 * hc.max()
co = {"atto twin of parent": d["atto_zp_z"], "nano twin of parent": d["nano_zp_z"], "solved parent (GN)": d["zg"],
      "trained parent (L1)": d["zL"], "atto twin of child": d[f"atto_zc{j}_z"], "nano twin of child": d[f"nano_zc{j}_z"]}
sel = np.where(onp | onc)[0]; rng = np.random.default_rng(0); sel = rng.permutation(sel)[:6000]
fig, axs = plt.subplots(2, 3, figsize=(16, 10))
print(f"{key} c{j}: positions parent-on {onp.sum()}, child-on {onc.sum()}, both {(onp & onc).sum()}")
print(f"{'coarse':22s} matched child effect (std units) per parent stratum, low -> high | slope ratio d/dchild : d/dparent")
for ax, (name, z) in zip(axs.flat, co.items()):
    t = gelu(z); s = t.std(); o = np.argsort(t[sel])
    ax.scatter(np.sqrt(hp[sel][o]), np.sqrt(hc[sel][o]), c=np.clip(t[sel][o] / s, -1, 6), s=4, cmap="viridis")
    ax.set_xlabel("sqrt parent"); ax.set_ylabel("sqrt child"); ax.set_title(name, fontsize=9)
    q = np.quantile(hp[onp], np.linspace(0, 1, 6)); eff = []
    for a in range(5):
        st = onp & (hp >= q[a]) & (hp <= q[a + 1]); A, B = t[st & onc], t[st & ~onc]
        eff.append((A.mean() - B.mean()) / s if len(A) > 15 and len(B) > 15 else np.nan)
    # local slopes (description only): coarse ~ parent + child on positions where either is on
    M = np.c_[hp[onp | onc], hc[onp | onc], np.ones((onp | onc).sum())]; cf = np.linalg.lstsq(M, t[onp | onc], rcond=None)[0]
    print(f"{name:22s} " + " ".join(f"{v:6.2f}" for v in eff) + f" | {cf[1] / cf[0]:6.2f}  (slopes per unit: parent {cf[0]:.3f} child {cf[1]:.3f})")
fig.suptitle(f"{key}: parent {tuple(int(v) for v in d['p'])} x child {tuple(int(v) for v in d['cs'][j])}; color = coarse neuron / std"); plt.tight_layout()
fn = os.path.join(FIG, f"scatter_{key}_c{j}.png"); plt.savefig(fn, dpi=75); print(fn)
