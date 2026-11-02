"""Decompose the Theorem 23 residual (no new generation): per layer; best c for fixed inputs (NNLS); how many true
training points carry theta. usage: t_decomp.py g3_all_3000.pt"""
import sys, numpy as np, torch, torch.nn.functional as F
from scipy.optimize import nnls
from shapes import DEV, LAM, Net
ck = torch.load("net_1000.pt"); net = Net().to(DEV); net.load_state_dict(ck["state"]); X, y = ck["X"].to(DEV), ck["y"].to(DEV)
names = [n for n, _ in net.named_parameters()]; ps = list(net.parameters()); th = torch.cat([p.detach().flatten() for p in ps])
def pair_grads(x, yy):
    """rows: (grad f_y - grad f_k)(x_s) for every s and k != y_s, flattened over all parameters"""
    rows, idx = [], []
    for s in range(len(x)):
        f = net(x[s:s + 1])[0]
        for k in range(4):
            if k == int(yy[s]): continue
            g = torch.autograd.grad(f[int(yy[s])] - f[k], ps, retain_graph=True); rows.append(torch.cat([a.flatten() for a in g])); idx.append((s, k))
    return torch.stack(rows), idx
sizes = [p.numel() for p in ps]; cuts = np.cumsum([0] + sizes)
def per_layer(v):
    return {n: float((th[a:b] - v[a:b]).norm() / th[a:b].norm()) for n, a, b in zip(names, cuts[:-1], cuts[1:])}
# true train set
Gt, it = pair_grads(X, y); p = F.softmax(net(X), 1).detach()
ct = torch.tensor([float(p[s, k]) / (len(X) * LAM) for s, k in it], device=DEV)
vt = ct @ Gt; print(f"true set: residual {float((th - vt).norm() / th.norm()):.4f}; per layer", {k: round(v, 3) for k, v in per_layer(vt).items()})
contrib = torch.zeros(len(X), device=DEV)
for (s, k), cc, row in zip(it, ct, Gt): contrib[s] += cc * row.norm()
order = torch.argsort(contrib, descending=True); u = (1 - (p ** 2).sum(1))
for n in [10, 30, 100, 300, 1000]:
    keep = set(order[:n].tolist()); m = torch.tensor([s in keep for s, _ in it], device=DEV)
    v = (ct * m) @ Gt; print(f"  top {n:4d} training points by contribution: residual {float((th - v).norm() / th.norm()):.4f}; their mean 1 - sum p^2 {float(u[order[:n]].mean()):.4f}")
print(f"  NNLS best c on the TRUE inputs: ", end="")
c_n, r = nnls(Gt.T.cpu().double().numpy(), th.cpu().double().numpy(), maxiter=20000); print(f"residual {r / float(th.norm()):.4f}")
# rebuilt candidates
d = torch.load(sys.argv[1]); xs, ys, cs = d["xs"].to(DEV), d["ys"].to(DEV), d["cs"].to(DEV)
Gs, iss = pair_grads(xs, ys); cg = torch.tensor([float(cs[s, k]) for s, k in iss], device=DEV); vs = cg @ Gs
print(f"rebuilt: residual with its own c {float((th - vs).norm() / th.norm()):.4f}; per layer", {k: round(v, 3) for k, v in per_layer(vs).items()})
c_s, r = nnls(Gs.T.cpu().double().numpy(), th.cpu().double().numpy(), maxiter=20000)
vb = torch.tensor(c_s, device=DEV, dtype=torch.float32) @ Gs
print(f"rebuilt: NNLS best c for the SAME inputs: residual {r / float(th.norm()):.4f}; per layer", {k: round(v, 3) for k, v in per_layer(vb).items()})
print(f"         nonzero c: {(c_s > 0).sum()} of {len(c_s)}")
