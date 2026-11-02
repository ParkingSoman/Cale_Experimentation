"""Generators for the I5 toy. None of them sees the truth: they read only the trained network (and LAM)."""
import torch, torch.nn.functional as F
from shapes import DEV, LAM, S

def balanced(m): return torch.arange(m, device=DEV) % 4

def g0_inversion(net, m, steps=256, seed=0):
    """Check E's rule: maximize the class logit (sigmoid pixels, small jitter)."""
    torch.manual_seed(seed); y = balanced(m); z = (0.01 * torch.randn(m, 1, S, S, device=DEV)).requires_grad_()
    opt = torch.optim.Adam([z], 0.05)
    for _ in range(steps):
        x = torch.sigmoid(z); dx, dy = torch.randint(-2, 3, (2,)).tolist(); x = torch.roll(x, (dy, dx), (2, 3))
        loss = -net(x).gather(1, y[:, None]).sum(); opt.zero_grad(); loss.backward(); opt.step()
    return torch.sigmoid(z).detach(), y

def g1_langevin(net, m, steps=1000, eta=1e-3, seed=0):
    """Lemma 19.1a with a flat prior on [0,1]^d: x ~ p_theta(y|x) (unadjusted Langevin, reflected into the box)."""
    torch.manual_seed(seed); y = balanced(m); x = torch.rand(m, 1, S, S, device=DEV)
    for _ in range(steps):
        x.requires_grad_(); lp = F.log_softmax(net(x), 1).gather(1, y[:, None]).sum(); g, = torch.autograd.grad(lp, x)
        x = (x.detach() + eta * g + (2 * eta) ** 0.5 * torch.randn_like(x)).clamp(0, 1)
    return x.detach(), y

def g2_stationary(net, m, steps=2000, lr=0.01, seed=0, log=None, a0=-7.0, x_init=None, fix_x=False):
    """Conjecture G: inputs x_s, labels y_s (balanced), weights c_s >= 0 with sum c_s = 1 (Theorem 2 is an AVERAGE) such that
    sum_s c_s grad l(x_s, y_s) = -LAM theta."""
    torch.manual_seed(seed); y = balanced(m)
    z = (0.1 * torch.randn(m, 1, S, S, device=DEV)) if x_init is None else torch.logit(x_init.clamp(1e-3, 1 - 1e-3))
    z = z.requires_grad_(not fix_x)
    a = torch.zeros(m, device=DEV, requires_grad=True)   # c = 1/m to start
    ps = list(net.parameters()); [p.requires_grad_(True) for p in ps]; th = [p.detach().clone() for p in ps]; nth = torch.sqrt(sum((t ** 2).sum() for t in th))
    opt = torch.optim.Adam([a] if fix_x else [z, a], lr)
    for it in range(steps):
        x = torch.sigmoid(z); c = torch.softmax(a, 0)
        l = (c * F.cross_entropy(net(x), y, reduction="none")).sum()
        g = torch.autograd.grad(l, ps, create_graph=True)
        R = torch.sqrt(sum(((gi + LAM * t) ** 2).sum() for gi, t in zip(g, th))) / (LAM * nth)
        opt.zero_grad(); R.backward(inputs=[a] if fix_x else [z, a]); opt.step()
        if log and it % 500 == 0: log(f"    g2 step {it}: relative residual {float(R):.4f}, sum c {float(c.sum()):.3f}")
    for p in ps: p.requires_grad_(False); p.grad = None
    return torch.sigmoid(z).detach(), y, torch.softmax(a, 0).detach()

def g3_kkt(net, m, steps=3000, lr=0.05, seed=0, log=None, layers=None, a0=-5.0):
    """Theorem 23: candidates x_s (labels balanced), free pairwise coefficients c_{s,k} >= 0 (k != y_s), such that
    sum c_{s,k} (grad f_{y_s} - grad f_k)(x_s) = theta (lam and N live in c). layers: list of parameter names to constrain (None = all)."""
    torch.manual_seed(seed); y = balanced(m); z = (0.1 * torch.randn(m, 1, S, S, device=DEV)).requires_grad_()
    a = torch.full((m, 4), a0, device=DEV, requires_grad=True)
    named = [(n, p) for n, p in net.named_parameters() if layers is None or n in layers]; ps = [p for _, p in named]
    [p.requires_grad_(True) for p in ps]; th = [p.detach().clone() for p in ps]; nth = torch.sqrt(sum((t ** 2).sum() for t in th))
    wrong = 1 - F.one_hot(y, 4).float(); opt = torch.optim.Adam([z, a], lr); hist = []   # target theta; c = p_k / (N lam) at truth
    for it in range(steps):
        x = torch.sigmoid(z); c = F.softplus(a) * wrong; f = net(x)
        s = (c * (f.gather(1, y[:, None]) - f)).sum()                          # sum c_{s,k} (f_y - f_k)
        g = torch.autograd.grad(s, ps, create_graph=True)
        R = torch.sqrt(sum(((t - gi) ** 2).sum() for gi, t in zip(g, th))) / nth
        opt.zero_grad(); R.backward(inputs=[z, a]); opt.step(); hist.append(float(R))
        if log and it % 500 == 0: log(f"    g3 step {it}: residual {float(R):.4f}, sum c {float(c.sum()):.3f}")
    for p in net.parameters(): p.requires_grad_(False); p.grad = None
    return torch.sigmoid(z).detach(), y, (F.softplus(a) * wrong).detach(), hist

def g4_varpro(net, m, steps=3000, K=100, lr=0.05, seed=0, log=None):
    """Statement 23.2: exact c >= 0 (NNLS) every K steps, Adam on the inputs only in between (Danskin)."""
    import numpy as np
    from scipy.optimize import nnls
    from torch.func import functional_call, vmap, jacrev
    torch.manual_seed(seed); y = balanced(m); z = (0.1 * torch.randn(m, 1, S, S, device=DEV)).requires_grad_()
    names = [n for n, _ in net.named_parameters()]; ps = list(net.parameters()); th = torch.cat([p.detach().flatten() for p in ps])
    params = {n: p.detach() for n, p in net.named_parameters()}; wrong = (1 - F.one_hot(y, 4).float()).bool()
    def logits1(pr, x1): return functional_call(net, pr, (x1[None],))[0]
    def rows(x):   # (m*3, |theta|): grad f_y - grad f_k for k != y
        out = []
        for i in range(0, m, 125):
            J = vmap(jacrev(logits1), in_dims=(None, 0))(params, x[i:i + 125])          # dict of (b, 4, *shape)
            Jf = torch.cat([J[n].reshape(J[n].shape[0], 4, -1) for n in names], 2)      # (b, 4, |theta|)
            yy = y[i:i + 125]; d = Jf.gather(1, yy[:, None, None].expand(-1, 1, Jf.shape[2])) - Jf
            out.append(d[wrong[i:i + 125]])
        return torch.cat(out)
    def solve_c(x):
        Gm = rows(x); c, _ = nnls(Gm.T.cpu().double().numpy(), th.cpu().double().numpy(), maxiter=50000)
        r = float(torch.sqrt(((th - torch.tensor(c, device=DEV, dtype=torch.float32) @ Gm) ** 2).sum()) / th.norm())
        cm = torch.zeros(m, 4, device=DEV); cm[wrong] = torch.tensor(c, device=DEV, dtype=torch.float32); return cm, r
    [p.requires_grad_(True) for p in ps]; opt = torch.optim.Adam([z], lr); hist = []
    for it in range(steps):
        if it % K == 0:
            c, rstar = solve_c(torch.sigmoid(z).detach()); hist.append(rstar)
            if log: log(f"    varpro step {it}: R* (exact c) {rstar:.4f}, nonzero c {int((c > 0).sum())}")
        x = torch.sigmoid(z); f = net(x); s = (c * (f.gather(1, y[:, None]) - f)).sum()
        g = torch.autograd.grad(s, ps, create_graph=True)
        R = torch.sqrt(((th - torch.cat([a.flatten() for a in g])) ** 2).sum()) / th.norm()
        opt.zero_grad(); R.backward(inputs=[z]); opt.step()
    c, rstar = solve_c(torch.sigmoid(z).detach()); hist.append(rstar)
    if log: log(f"    varpro final: R* {rstar:.4f}")
    for p in net.parameters(): p.requires_grad_(False); p.grad = None
    return torch.sigmoid(z).detach(), y, c, hist

def g5_boundary(net, m, steps=256, lr=0.05, seed=0):
    """E7 on the toy: cross-entropy to 1/2 e_a + 1/2 e_b, b = a's nearest class by fc weight rows."""
    torch.manual_seed(seed); a = balanced(m); W = net.fc.weight.detach(); Wn = W / W.norm(dim=1, keepdim=True); Sm = Wn @ Wn.T
    Sm.fill_diagonal_(-2); b = Sm.argmax(1)[a]; z = (0.01 * torch.randn(m, 1, S, S, device=DEV)).requires_grad_(); opt = torch.optim.Adam([z], lr)
    for _ in range(steps):
        x = torch.sigmoid(z); dx, dy = torch.randint(-2, 3, (2,)).tolist(); x = torch.roll(x, (dy, dx), (2, 3))
        lp = F.log_softmax(net(x), 1); loss = -0.5 * (lp.gather(1, a[:, None]) + lp.gather(1, b[:, None])).sum(); opt.zero_grad(); loss.backward(); opt.step()
    return torch.sigmoid(z).detach(), a, b
