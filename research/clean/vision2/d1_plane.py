"""D1 step 1: the family plane. x = parent, y = child (post-GELU, model units). Color = mean of a coarse neuron in each
cell (atto twin, nano twin, solved parent, trained parent), z-scored. Cells with < 30 positions are blank.
usage: python d1_plane.py KEY [child index]"""
import os, sys
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import erf
HF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hf"); D = os.path.join(HF, "imagenet", "d1")
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "d"); os.makedirs(FIG, exist_ok=True)
gelu = lambda z: 0.5 * z * (1 + erf(z / np.sqrt(2)))
key = sys.argv[1]; j = int(sys.argv[2]) if len(sys.argv) > 2 else 0; d = np.load(os.path.join(D, key + ".npz"))
hp, hc = gelu(d["zp"]), gelu(d[f"zc{j}"])
coarse = {"atto twin of parent": gelu(d["atto_zp_z"]), "nano twin of parent": gelu(d["nano_zp_z"]),
          "solved parent (GN)": gelu(d["zg"]), "trained parent (L1)": gelu(d["zL"]),
          "atto twin of child": gelu(d[f"atto_zc{j}_z"]), "nano twin of child": gelu(d[f"nano_zc{j}_z"])}
def edges(v, k=14):            # bins: one for "off" (< 2% of max), the rest quantiles of the on part
    on = v[v > 0.02 * v.max()]; return np.unique(np.r_[v.min() - 1e-6, 0.02 * v.max(), np.quantile(on, np.linspace(0, 1, k))[1:-1], v.max() + 1e-6])
ex, ey = edges(hp), edges(hc); ix, iy = np.digitize(hp, ex) - 1, np.digitize(hc, ey) - 1
fig, axs = plt.subplots(2, 3, figsize=(16, 10)); rows = []
for ax, (name, t) in zip(axs.flat, coarse.items()):
    tz = (t - t.mean()) / (t.std() + 1e-9); M = np.full((len(ey) - 1, len(ex) - 1), np.nan); N = np.zeros_like(M)
    for a in range(len(ex) - 1):
        for b in range(len(ey) - 1):
            s = (ix == a) & (iy == b); N[b, a] = s.sum()
            if s.sum() >= 30: M[b, a] = tz[s].mean()
    im = ax.imshow(M, origin="lower", cmap="RdBu_r", vmin=-np.nanmax(np.abs(M)), vmax=np.nanmax(np.abs(M)), aspect="auto")
    ax.set_xticks(range(len(ex) - 1)); ax.set_xticklabels([f"{v:.1f}" for v in ex[1:]], rotation=90, fontsize=6)
    ax.set_yticks(range(len(ey) - 1)); ax.set_yticklabels([f"{v:.1f}" for v in ey[1:]], fontsize=6)
    ax.set_xlabel("parent (bin top)"); ax.set_ylabel("child (bin top)"); ax.set_title(name, fontsize=9); plt.colorbar(im, ax=ax)
    # summary: mean coarse (z) in the four corners: off/off, parent-only top, child-only top, both top
    q = lambda v: v > np.quantile(v[v > 0.02 * v.max()], .8) if (v > 0.02 * v.max()).sum() > 50 else v > v.max() * .5
    P, C, P0, C0 = q(hp), q(hc), hp < 0.02 * hp.max(), hc < 0.02 * hc.max()
    rows.append((name, *[float(tz[s].mean()) if s.sum() > 20 else np.nan for s in (P0 & C0, P & C0, P0 & C, P & C)], int((P0 & C).sum()), int((P & C).sum())))
fig.suptitle(f"{key}: parent {tuple(d['p'])} x child {tuple(d['cs'][j])}  (color = coarse neuron, z-scored cell mean)"); plt.tight_layout()
fn = os.path.join(FIG, f"plane_{key}_c{j}.png"); plt.savefig(fn, dpi=80); print(fn)
print(f"{'coarse neuron':24s} {'off/off':>8s} {'par-only':>8s} {'chi-only':>8s} {'both':>8s}  n(chi-only) n(both)")
for r in rows: print(f"{r[0]:24s} " + " ".join(f"{v:8.2f}" for v in r[1:5]) + f"  {r[5]:6d} {r[6]:6d}")
print("corr parent-child %.3f; child on-rate %.3f; parent on-rate %.3f; P(parent on | child on) %.3f" % (
    np.corrcoef(hp, hc)[0, 1], (hc > .02 * hc.max()).mean(), (hp > .02 * hp.max()).mean(), (hp[hc > .02 * hc.max()] > .02 * hp.max()).mean()))
