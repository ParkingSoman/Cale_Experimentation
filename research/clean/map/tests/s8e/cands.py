"""Per target: each candidate's battery profile correlation with the target, and its top/bottom stimuli."""
import sys, os, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(H, "battery.json"))); L = D["labels"]
C = json.load(open(os.path.join(H, "..", "..", "s8", "candidates.json")))
for t in sys.argv[1:]:
    tk = t.replace("_", ","); T = np.array(D["resp"][tk]["max"]); Tn = np.array(D["resp"][tk]["min"])
    print(f"==== target {tk}")
    for k, (b, n) in C[t].items():
        M = np.array(D["resp"][f"{b},{n}"]["max"]); N = np.array(D["resp"][f"{b},{n}"]["min"])
        r = np.corrcoef(T, M)[0, 1]; rn = np.corrcoef(T, N)[0, 1]
        o = np.argsort(-M)[:7]; u = np.argsort(N)[:4]
        print(f" {k} {b},{n}: r(max)={r:+.2f} r(min)={rn:+.2f} | top: " + ", ".join(f"{L[i]} {M[i]:.2f}" for i in o) + " | bot: " + ", ".join(f"{L[i]} {N[i]:.2f}" for i in u))
