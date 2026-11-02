"""N1, N3: Neural Feature Ansatz. Gram of each reading layer's 1x1 input convs vs AGOP / Fisher / plain moment at its input."""
import numpy as np, torch
from inception import *
from family import Data, pool3
import sys
m = load_model(); D = Data(seed=0); KIND = sys.argv[1] if len(sys.argv) > 1 else "real"; NIMG = 500 if KIND == "real" else 250
idx = np.sort(np.random.default_rng(61).choice(D.tr, NIMG, replace=False)); NV = 4; rng = np.random.default_rng(62)
def images(b):
    X = np.asarray(D.X[b])
    if KIND == "noise": return rng.integers(0, 256, X.shape, dtype=np.uint8)
    if KIND == "shuffled":
        Y = X.reshape(len(X), -1, 3).copy()
        for j in range(len(Y)): Y[j] = Y[j][rng.permutation(Y.shape[1])]
        return Y.reshape(X.shape)
    return X
SITES = {"3a->3b": "mixed3b", "3b->4a": "mixed4a", "4a->4b": "mixed4b", "4d->4e": "mixed4e"}
def gram(block):
    Ws = [getattr(m, f"{block}_{b}_pre_relu_conv").weight.detach()[:, :, 0, 0].cpu().double() for b in ["1x1", "3x3_bottleneck", "5x5_bottleneck", "pool_reduce"]]
    return sum(W.T @ W for W in Ws)
def forward(x):
    a3 = to_3a(m, x).detach().requires_grad_(); h3b = F.relu(pre_3b(m, a3)); xb = pool3(h3b); xb.retain_grad()
    h4a = mixed(m, "mixed4a", xb); h4a.retain_grad(); h = h4a
    for n in ["mixed4b", "mixed4c", "mixed4d"]: h = mixed(m, n, h)
    h4d = h; h4d.retain_grad(); h = pool3(mixed(m, "mixed4e", h4d))
    for n in ["mixed5a", "mixed5b"]: h = mixed(m, n, h)
    return m.softmax2_pre_activation_matmul(F.avg_pool2d(h, 7).reshape(-1, 1024)), {"3a->3b": a3, "3b->4a": xb, "4a->4b": h4a, "4d->4e": h4d}
acc = {s: {k: [0, 0] for k in ["agop", "fisher", "plain"]} for s in SITES}
gen = torch.Generator().manual_seed(0)
for i in range(0, len(idx), 25):
    half = int(i >= len(idx) // 2); x = prep(images(idx[i:i + 25]))
    for r in range(NV):
        lg, acts = forward(x)
        if r == 0:
            for s, a in acts.items():
                A = a.detach().permute(0, 2, 3, 1).reshape(-1, a.shape[1]).cpu().double(); acc[s]["plain"][half] = acc[s]["plain"][half] + A.T @ A
        v = torch.randn(lg.shape, generator=gen).to(DEV); (lg * v).sum().backward(retain_graph=True)
        for s, a in acts.items():
            g = a.grad.permute(0, 2, 3, 1).reshape(-1, a.shape[1]).cpu().double(); acc[s]["agop"][half] = acc[s]["agop"][half] + g.T @ g / NV; a.grad = None
        lp = F.log_softmax(lg, 1); top = lp.detach().exp().topk(10, 1)
        eps = (torch.randint(0, 2, top.values.shape, generator=gen) * 2 - 1).float().to(DEV)
        (eps * top.values.sqrt() * lp.gather(1, top.indices)).sum().backward()
        for s, a in acts.items():
            g = a.grad.permute(0, 2, 3, 1).reshape(-1, a.shape[1]).cpu().double(); acc[s]["fisher"][half] = acc[s]["fisher"][half] + g.T @ g / NV; a.grad = None
    print(f"  {i + 25}/{len(idx)}", flush=True) if (i // 25) % 5 == 4 else None
def cos(A, B): return float((A * B).sum() / A.norm() / B.norm())
def pw(A, a):
    e, U = torch.linalg.eigh(A); return U @ torch.diag(e.clamp(min=0) ** a) @ U.T
print("site   | n   | rel agop | rel fisher | cos(G, agop^.5) | cos(G, agop) | cos(G, fisher^.5) | cos(G, fisher) | cos(G, plain^.5) | cos(G, plain) | cos(G, I)")
for s, blk in SITES.items():
    G = gram(blk); n = G.shape[0]; T = {k: v[0] + v[1] for k, v in acc[s].items()}
    rel = lambda k: cos(acc[s][k][0], acc[s][k][1])
    print(f"{s:6s} | {n:3d} | {rel('agop'):.3f} | {rel('fisher'):.3f} | {cos(G, pw(T['agop'], .5)):.3f} | {cos(G, T['agop']):.3f} | "
          f"{cos(G, pw(T['fisher'], .5)):.3f} | {cos(G, T['fisher']):.3f} | {cos(G, pw(T['plain'], .5)):.3f} | {cos(G, T['plain']):.3f} | {cos(G, torch.eye(n, dtype=G.dtype)):.3f}")
torch.save({s: {k: v[0] + v[1] for k, v in acc[s].items()} for s in SITES}, "cache/n1_nfa.pt" if KIND == "real" else f"cache/n1_nfa_{KIND}.pt")
