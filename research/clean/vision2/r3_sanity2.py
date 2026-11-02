import numpy as np, sys
exec(open("r3_treefold.py").read().split("Hz = (H16")[0])
SAL = [97, 159]; mask = np.ones(768, bool); mask[SAL] = False
sal_share = (D17[SAL] ** 2).sum(0) / (D17 ** 2).sum(0)
en_all = np.linalg.norm(H17, axis=0) * np.linalg.norm(D17, axis=0)
en_ns = np.linalg.norm(H17, axis=0) * np.linalg.norm(D17[mask], axis=0)
top = np.argsort(-en_all)[:384]
print("median share of write on saliency coords: all %.3f, top-384 by energy %.3f" % (np.median(sal_share), np.median(sal_share[top])))
for name, en in (("energy_all", en_all), ("energy_no_saliency", en_ns)):
    order = np.argsort(-en); row = []
    for K in (2304, 1536, 768, 384, 96, 16):
        keep = np.sort(order[:K]); Dk, bk = ls(H17[:, keep], Y); row.append("%d:%.4f" % (K, evaluate(Dk, bk, keep, W16).mean()))
    print(name, " ".join(row), flush=True)
