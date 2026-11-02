"""N5: S_l = off-diagonal cos(Gram of layer l's input convs, AGOP of the logits at layer l's input) per input set."""
import numpy as np, torch
from inception import *
from family import Data, pool3
from scipy.special import softmax
m = load_model(); D = Data(seed=0)
for p in m.parameters(): p.requires_grad_(False)
def gram(block):
    Ws = [getattr(m, f"{block}_{b}_pre_relu_conv").weight.detach()[:, :, 0, 0].cpu().double() for b in ["1x1", "3x3_bottleneck", "5x5_bottleneck", "pool_reduce"]]
    return sum(W.T @ W for W in Ws)
G = {"3b->4a": gram("mixed4a"), "4a->4b": gram("mixed4b"), "4d->4e": gram("mixed4e")}
def agop(X, nv=4):
    gen = torch.Generator().manual_seed(0); acc = {k: 0 for k in G}
    for i in range(0, len(X), 16):
        x = prep(X[i:i + 16])
        with torch.no_grad(): a3 = to_3a(m, x); h3b = F.relu(pre_3b(m, a3))
        xb = pool3(h3b).detach().requires_grad_(); h4a = mixed(m, "mixed4a", xb); h4a.retain_grad(); h = h4a
        for n in ["mixed4b", "mixed4c", "mixed4d"]: h = mixed(m, n, h)
        h4d = h; h4d.retain_grad(); h = pool3(mixed(m, "mixed4e", h4d))
        for n in ["mixed5a", "mixed5b"]: h = mixed(m, n, h)
        lg = m.softmax2_pre_activation_matmul(F.avg_pool2d(h, 7).reshape(-1, 1024))
        for r in range(nv):
            v = torch.randn(lg.shape, generator=gen).to(DEV); (lg * v).sum().backward(retain_graph=r < nv - 1)
            for k, a in [("3b->4a", xb), ("4a->4b", h4a), ("4d->4e", h4d)]:
                g = a.grad.permute(0, 2, 3, 1).reshape(-1, a.shape[1]).cpu().double(); acc[k] = acc[k] + g.T @ g; a.grad = None
    return acc
def off(A, B):
    mk = ~torch.eye(A.shape[0], dtype=bool); a, b = A[mk], B[mk]; return float((a * b).sum() / a.norm() / b.norm())
P = softmax(np.asarray(D.T[D.tr], np.float64), 1); uns = D.tr[1 - (P ** 2).sum(1) > .1]
sets = {"real unsure": np.asarray(D.X[np.sort(np.random.default_rng(141).choice(uns, 128, replace=False))]),
        "class inversions": np.load("cache/e1_self_256.npy")[:128], "boundary": np.load("cache/e7_bnd_1024_128_0.npy")[:128],
        "reader pictures": np.load("cache/c31_fv.npy")[:104]}
print("set              | S 3b->4a | S 4a->4b | S 4d->4e")
for k, X in sets.items():
    A = agop(X); print(f"{k:16s} | {off(G['3b->4a'], A['3b->4a']):.3f}   | {off(G['4a->4b'], A['4a->4b']):.3f}   | {off(G['4d->4e'], A['4d->4e']):.3f}", flush=True)
