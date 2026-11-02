"""C1b (Part VIII, after C1): which objective tracks the true KL along the exact-uniform PCG path?"""
import numpy as np, torch, time
import sys; sys.argv = ["x", "photos", "2000", "photo2000"]
SRC = open("c50_exactcg.py").read(); exec(SRC.split("# self-check of Prop. 31.2")[0])
perm = np.random.default_rng(50).permutation(D.tr); Iho = np.sort(perm[2000:3000]); Xho = D.X[Iho]
def Lclass(V):
    L = 0.0
    for i in range(0, 1000, 10):
        Gm, _, _ = gammas(Iho[i:i + 10]); L += float(((Gm * V).sum((2, 3)) ** 2).sum()) / 2
    return L / 1000
Vk = [Vs["S"] * mask] + [torch.load(f"cache/c50_photo2000_it{k}_V.pt").to(DEV) for k in range(1, 16)]
print(f"{'it':>3s} {'true KL':>9s} {'L_class':>9s} {'L_uniform':>10s}")
for k, V in enumerate(Vk):
    kl = per_image(build(V), Iho).mean(); print(f"{k:3d} {kl:9.5f} {Lclass(V):9.5f} {Fv_exact(V, Xho)[1]:10.4g}  ({time.time() - t0:.0f}s)", flush=True)
print("done")
