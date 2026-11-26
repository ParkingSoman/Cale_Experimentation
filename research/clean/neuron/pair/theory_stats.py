"""theory agent, step 1: per-Base-neuron stats for stage-4 block 0, from cached data only.

match   = best test corr (after GELU, all positions) of the Base neuron to any Small neuron   (grams test)
std     = output std after GELU (test)
wnorm   = norm of its class-score write (writes_V b0)
value   = std x wnorm  (how much it moves the class scores, on average)
freq    = fraction of the 3,000 fit photos where its peak pre-activation is > 0  (tools_peaks_3000)
redund  = best test corr to another Base neuron
Saves results/pair/theory/stats.npz (small).
"""
import os
import numpy as np
from scipy.stats import spearmanr

R = "../results/pair/"
O = R + "theory/"
g = np.load(R + "grams.npz")
out = {}
for split in ("fit", "test"):
    Cbb, Cbs, vs = g[f"{split}_Cbb"].astype(np.float64), g[f"{split}_Cbs"].astype(np.float64), g[f"{split}_vs"]
    sb = np.sqrt(np.clip(np.diag(Cbb), 1e-12, None)); ss = np.sqrt(np.clip(vs, 1e-12, None))
    Rbs = Cbs / sb[:, None] / ss[None, :]
    Rbb = Cbb / sb[:, None] / sb[None, :]
    np.fill_diagonal(Rbb, -1)
    out[f"{split}_match"] = Rbs.max(1); out[f"{split}_best_s"] = Rbs.argmax(1)
    out[f"{split}_sbest"] = Rbs.max(0); out[f"{split}_sbest_b"] = Rbs.argmax(0)
    out[f"{split}_redund"] = Rbb.max(1); out[f"{split}_redund_b"] = Rbb.argmax(1)
    out[f"{split}_std"] = sb; out[f"{split}_sstd"] = ss
    out[f"{split}_mean"] = g[f"{split}_mb"]
V = np.load(R + "writes_V.npz")
out["wnorm"] = np.linalg.norm(V["b0"], axis=1); out["swnorm"] = np.linalg.norm(V["s0"], axis=1)
pk = np.load(R + "tools_peaks_3000.npz")
out["freq"] = (pk["base"][:, 0, :].astype(np.float32) > 0).mean(0)
out["sfreq"] = (pk["small"][:, 0, :].astype(np.float32) > 0).mean(0)
out["value"] = out["test_std"] * out["wnorm"]
np.savez(O + "stats.npz", **out)
m = out["test_match"]
print("match: mean %.3f median %.3f  >.7: %.3f  <.4: %.3f" % (m.mean(), np.median(m), (m > .7).mean(), (m < .4).mean()))
print("fit-test agreement of match: r=%.3f" % np.corrcoef(out["fit_match"], m)[0, 1])
print("Small neurons claimed as best by >=1 Base neuron:", len(set(out["test_best_s"])))
for k in ("freq", "test_std", "wnorm", "value", "test_redund", "test_mean"):
    print(f"spearman(match, {k}) = {spearmanr(m, out[k])[0]:.3f}")
