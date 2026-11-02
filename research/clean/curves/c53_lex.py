"""C3 (Part VIII, 31.8): along the cached gate-free path, does the exact-gate objective L_ex track the true KL?"""
import numpy as np, torch, time, sys
sys.argv = ["x", "photos", "300", "lex"]
SRC = open("c50_exactcg.py").read(); exec(SRC.split("# self-check of Prop. 31.2")[0])
perm = np.random.default_rng(50).permutation(D.tr); Iho = np.sort(perm[2000:2300]); Xho = D.X[Iho]
def Lex(V, X, bs=10):
    L = 0.0
    for i in range(0, len(X), bs):
        x, p = fin(X[i:i + bs]); o = torch.ones_like(x[:, :1]); y = torch.cat([x, o, p, o], 1).flatten(2)
        with torch.no_grad():
            pre = [F.conv2d(inp, getattr(m, n).weight, getattr(m, n).bias) for n, inp in zip(IN4A, [x, x, x, p])]
            a = torch.cat([t.flatten(2) for t in pre], 1); e = torch.einsum("jk,bkp->bjp", V * mask, y)
            dl = F.relu(a - e) - F.relu(a)                                                   # exact reader change
        post = [F.relu(t) for t in pre]; pp = [t.clone().requires_grad_(True) for t in post]; lg = upper(*pp)
        w = torch.zeros_like(lg, requires_grad=True); gw = torch.autograd.grad((lg * w).sum(), pp, create_graph=True)
        tans = [t.reshape(len(x), -1, 14, 14) for t in torch.split(dl, sizes, 1)]
        u = torch.autograd.grad(sum((g_ * t).sum() for g_, t in zip(gw, tans)), w)[0].detach(); u = u - u.mean(1, keepdim=True)
        L += 0.5 * float((u ** 2).sum())
    return L / len(X)
Vk = [Vs["S"] * mask] + [torch.load(f"cache/c50_photo2000_it{k}_V.pt").to(DEV) for k in range(1, 16)]
print(f"{'it':>3s} {'true KL':>9s} {'L_ex':>10s} {'L_gatefree':>10s}")
for k, V in enumerate(Vk):
    print(f"{k:3d} {per_image(build(V), Iho).mean():9.5f} {Lex(V, Xho):10.4g} {Fv_exact(V, Xho)[1]:10.4g}  ({time.time() - t0:.0f}s)", flush=True)
print("done")
