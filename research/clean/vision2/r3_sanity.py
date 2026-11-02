import numpy as np
exec(open("r3_treefold.py").read().split("Hz = (H16")[0])
en = np.linalg.norm(H17, axis=0) * np.linalg.norm(D17, axis=0); order = np.argsort(-en)
for K in (3072, 2304, 1536, 768, 384, 96, 16, 1):
    keep, rem = np.sort(order[:K]), np.sort(order[K:])
    raw = evaluate(D17[:, keep], g17 * c17, keep, W16).mean()
    Dk, bk = ls(H17[:, keep], Y); lsacc = evaluate(Dk, bk, keep, W16).mean()
    print(K, "cut no-refit no-bias %.4f" % raw, "cut_ls %.4f" % lsacc, flush=True)
