"""S: support of theta with free c. NNLS restricted to subsets of the true training set (no generation)."""
import numpy as np, torch, torch.nn.functional as F
from scipy.optimize import nnls
from shapes import DEV, LAM, Net
ck = torch.load("net_1000.pt"); net = Net().to(DEV); net.load_state_dict(ck["state"]); X, y = ck["X"].to(DEV), ck["y"].to(DEV)
ps = list(net.parameters()); th = torch.cat([p.detach().flatten() for p in ps]).cpu().double().numpy(); nth = np.linalg.norm(th)
rows, own = [], []
for s in range(len(X)):
    f = net(X[s:s + 1])[0]
    for k in range(4):
        if k == int(y[s]): continue
        g = torch.autograd.grad(f[int(y[s])] - f[k], ps, retain_graph=True); rows.append(torch.cat([a.flatten() for a in g]).cpu().double().numpy()); own.append(s)
G = np.stack(rows); own = np.array(own); p = F.softmax(net(X), 1).detach().cpu().numpy(); u = 1 - (p ** 2).sum(1)
ct = np.array([p[s, k] for s in range(len(X)) for k in range(4) if k != int(y[s])]) / (len(X) * LAM)
contrib = np.zeros(len(X)); np.add.at(contrib, own, ct * np.linalg.norm(G, axis=1)); order = np.argsort(-contrib)
def R(idx):
    m = np.isin(own, idx); c, r = nnls(G[m].T, th, maxiter=50000); return r / nth
rng = np.random.default_rng(0); unsure = np.where(u > .1)[0]; sure = np.where(u <= .1)[0]
print(f"all 1000: {R(np.arange(1000)):.4f} | unsure only ({len(unsure)}): {R(unsure):.4f} | sure only ({len(sure)}): {R(sure):.4f}")
for n in [len(unsure), 10, 30, 100, 300]:
    rr = [R(rng.choice(1000, n, replace=False)) for _ in range(5)]
    print(f"  n = {n:4d}: top contributors {R(order[:n]):.4f} | random {np.mean(rr):.4f} (sd {np.std(rr):.4f})"
          + (f" | unsure {R(unsure):.4f}" if n == len(unsure) else ""))
