"""Descriptive: is a child's top upstream contributor the same neuron across its strongest tokens, or does it change
with context? Per token attribution from layers L-6..L-1 (same formula as tf_diag), child's top 0.5% tokens."""
import json, os, sys
import numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tf_diag.py")).read().split("blk = np.repeat")[0])
rng = np.random.default_rng(0); sample = rng.choice(rem, 300, replace=False)
modal, distinct, share_top_mass = [], [], []
for c in sample:
    t = np.argpartition(-np.abs(H[:, c]), ntop)[:ntop]
    A = MG[c] * ((dsilu(Zg[t, c]) * Zu[t, c])[:, None] * Pr[t]) + MU[c] * (silu(Zg[t, c])[:, None] * Pr[t])   # tokens x parents
    top = np.abs(A).argmax(1)
    modal.append(np.bincount(top).max() / len(t)); distinct.append(len(set(top)))
    share_top_mass.append(float(np.median(np.abs(A).max(1) / np.abs(A).sum(1))))
out = {"L": L, "n_tokens_per_child": int(ntop), "modal_parent_share_median": float(np.median(modal)),
       "distinct_top_parents_median": float(np.median(distinct)), "top_parent_share_of_attr_mass_median": float(np.median(share_top_mass))}
print(out)
p = os.path.join(ROOT, "results", "clean-tf-ctx.json"); res = json.load(open(p)) if os.path.exists(p) else {}
res[f"L{L}"] = out; json.dump(res, open(p, "w"), indent=1)
