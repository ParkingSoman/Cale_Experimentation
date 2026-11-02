"""Image-level containment on the 20k cache (blocks 11-17): P(cand image-max > its 99th pct | target image in its top 200),
lift over 1%, class-mean correlation, and image breadth P(image-max > 0.5 * 99.9th pct)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import last_lib as L
C = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "s8", "candidates.json")))
for key in sys.argv[1:]:
    tb, tn = map(int, key.split("_")); t = L.amax(tb, tn); top = np.argsort(-t)[:200]; ct = L.class_mean(t)
    br = lambda v: float((v > .5 * np.percentile(v, 99.9)).mean())
    print(f"{key}: target image breadth {br(t):.4f}")
    for c, (b, n) in C[key].items():
        if b < 11: print(f"  {c} b{b}n{n}: not in cache"); continue
        v = L.amax(b, n); thr = np.percentile(v, 99); p = (v[top] > thr).mean(); thr9 = np.percentile(v, 90); p9 = (v[top] > thr9).mean()
        cc = np.corrcoef(ct, L.class_mean(v))[0, 1]
        print(f"  {c} b{b}n{n}: P(>p99|top200) {p:.2f} (lift {p/.01:5.1f})  P(>p90) {p9:.2f}  class corr {cc:+.2f}  breadth {br(v):.4f}")
