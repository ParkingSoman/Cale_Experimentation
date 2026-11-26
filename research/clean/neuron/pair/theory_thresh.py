"""theory agent, step 4: is coarsening a lowered threshold on the counterpart's PRE-activation?

Data: per-photo peak pre-activations, block 0, 3,000 photos (tools_peaks_3000). Pairs: the 2,017 agreeing (s, p).
For each pair, s's on-set = photos where s's peak > 0. Compare ways to predict that set, each picking the SAME number
of photos as s fires on (so only ranking matters):
  p_rank    rank photos by p's peak pre-activation (= p with a lowered threshold)
  q_rank    same with a look-alike Base neuron q (the next-best Base match of s by peak corr)   [control]
  shuf      p's ranking with photos shuffled                                              [control, chance]
Score: recall of s's on-set. Also: p's own on-set recall, and for s-on-p-off photos the percentile of p's peak
within p's off photos (threshold story predicts high, near 1).
Fit/test: none needed (no fitted parameters except the matched count).
"""
import json
import numpy as np

R = "../results/pair/"
pk = np.load(R + "tools_peaks_3000.npz")
S = pk["small"][:, 0, :].astype(np.float32); B = pk["base"][:, 0, :].astype(np.float32)
st = np.load(R + "stream_cov.npz"); agree, pmap = st["agree"], st["pmap"]
Rp = np.load(R + "tools_peakR_b0.npy").astype(np.float32)  # (3072, 4096)
rng = np.random.default_rng(0)
out = []
for s in agree:
    p = pmap[s]
    on = S[:, s] > 0; k = on.sum()
    if k < 10:
        continue
    r = Rp[s].copy(); r[p] = -9; q = r.argmax()
    def rec(x):
        top = np.argsort(-x)[:k]; return on[top].mean()
    pon = B[:, p] > 0
    sp = on & ~pon; off = ~pon
    pct = np.mean([(B[off, p] < v).mean() for v in B[sp, p]]) if sp.any() else np.nan
    out.append(dict(s=int(s), p=int(p), k=int(k), kp=int(pon.sum()), p_own_recall=float(on[pon].mean() / 1) if pon.any() else 0,
                    p_own_cover=float((on & pon).sum() / k), p_rank=float(rec(B[:, p])), q_rank=float(rec(B[:, q])),
                    shuf=float(rec(rng.permutation(B[:, p]))), pct_off=float(pct)))
A = {k: np.array([o[k] for o in out]) for k in out[0]}
print(f"pairs {len(out)}; s fires on median {np.median(A['k'])} photos, p on {np.median(A['kp'])}")
print(f"p's own on-set covers {np.median(A['p_own_cover']):.3f} of s's on-set (median)")
for k in ("p_rank", "q_rank", "shuf"):
    print(f"recall of s's on-set, same count, by {k}: median {np.median(A[k]):.3f} mean {A[k].mean():.3f}")
print(f"s-on p-off photos: p's peak is at percentile {np.nanmedian(A['pct_off']):.3f} of p's off photos (chance .5)")
json.dump(out, open(R + "theory/thresh.json", "w"))
