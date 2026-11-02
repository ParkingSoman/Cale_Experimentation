"""I5 toy: a known input distribution (4 shape classes) and a small CNN trained to near-stationarity with weight decay."""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
S = 32; LAM = 1e-3

def sample(n, seed):
    """Truth: class c in {disc, square, triangle, ring}; random centre, size, brightness, background, noise. Values in [0, 1]."""
    r = np.random.default_rng(seed); yy, xx = np.mgrid[0:S, 0:S] + 0.5; X = np.zeros((n, S, S), np.float32); y = r.integers(0, 4, n)
    for i in range(n):
        cx, cy = r.uniform(9, 23, 2); s = r.uniform(4, 9); bg = r.uniform(0, .4); fg = np.clip(bg + r.choice([-1, 1]) * r.uniform(.3, .6), 0, 1)
        dx, dy = xx - cx, yy - cy
        if y[i] == 0: m = dx ** 2 + dy ** 2 < s ** 2
        elif y[i] == 1: m = (abs(dx) < s * .85) & (abs(dy) < s * .85)
        elif y[i] == 2: m = (dy < s * .8) & (dy > -s) & (abs(dx) < (dy + s) * .6)
        else: d = np.sqrt(dx ** 2 + dy ** 2); m = (d < s) & (d > s * .55)
        X[i] = np.where(m, fg, bg) + r.normal(0, .05, (S, S))
    return torch.tensor(np.clip(X, 0, 1))[:, None].to(DEV), torch.tensor(y).to(DEV)

class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.c1 = nn.Conv2d(1, 16, 5, padding=2); self.c2 = nn.Conv2d(16, 32, 5, padding=2); self.fc = nn.Linear(32 * 8 * 8, 4)
    def feats(self, x):   # conv2 output after ReLU and pool (B, 32, 8, 8)
        mp = lambda t: F.max_pool2d(t, 2, return_indices=True)[0]   # same forward; twice differentiable (needed by G2)
        return mp(F.relu(self.c2(mp(F.relu(self.c1(x))))))
    def forward(self, x): return self.fc(self.feats(x).flatten(1))

def objective(net, X, y):
    return F.cross_entropy(net(X), y) + LAM / 2 * sum((p ** 2).sum() for p in net.parameters())

def residual(net, X, y):
    """|| grad of the regularised loss || / || lam theta ||: 0 at an exact stationary point (Theorem 2's premise)."""
    g = torch.autograd.grad(objective(net, X, y), list(net.parameters()))
    return float(torch.sqrt(sum((a ** 2).sum() for a in g)) / (LAM * torch.sqrt(sum((p ** 2).sum() for p in net.parameters()))))

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    torch.manual_seed(0); X, y = sample(N, 0); Xt, yt = sample(5000, 1); net = Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for it in range(4000):
        opt.zero_grad(); objective(net, X, y).backward(); opt.step()
        if it % 1000 == 999: print(f"adam {it + 1}: obj {float(objective(net, X, y).detach()):.4f} residual {residual(net, X, y):.3f}", flush=True)
    opt = torch.optim.LBFGS(net.parameters(), lr=1, max_iter=50, history_size=50, line_search_fn="strong_wolfe")
    for it in range(40):
        def cl():
            opt.zero_grad(); l = objective(net, X, y); l.backward(); return l
        opt.step(cl)
    with torch.no_grad():
        acc = float((net(X).argmax(1) == y).float().mean()); acct = float((net(Xt).argmax(1) == yt).float().mean())
        p = F.softmax(net(Xt), 1); u = float((1 - (p ** 2).sum(1)).mean()); ptr = F.softmax(net(X), 1); utr = float((1 - (ptr ** 2).sum(1)).mean())
    print(f"final: residual {residual(net, X, y):.4f} | train acc {acc:.3f} test acc {acct:.3f} | 1-sum p^2 train {utr:.3f} test {u:.3f}")
    # Theorem 1 balance for conv1 and conv2 channels
    W1, W2 = net.c1.weight.detach(), net.c2.weight.detach()
    in1 = (W1 ** 2).flatten(1).sum(1) + net.c1.bias.detach() ** 2; out1 = (W2 ** 2).sum((0, 2, 3))
    in2 = (W2 ** 2).flatten(1).sum(1) + net.c2.bias.detach() ** 2; out2 = (net.fc.weight.detach().reshape(4, 32, 64) ** 2).sum((0, 2))
    print(f"balance in^2/out^2 median: conv1 {float((in1 / out1).median()):.3f}, conv2 {float((in2 / out2).median()):.3f}")
    torch.save(dict(state={k: v.cpu() for k, v in net.state_dict().items()}, X=X.cpu(), y=y.cpu(), N=N), f"net_{N}.pt")
