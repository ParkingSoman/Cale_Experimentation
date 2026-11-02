# serves: TV15
"""Synthetic TRAINED pure-stream networks (Definition DV1): how tight is Theorem TV15 against Theorem TV6 on a network
that has learned something? Not InceptionV1; numbers go to the Appendix record of Part B.V.
Task: 4 classes of points in R^8 (a nonlinear rule), first write e(x) = A x, three blocks of 32 neurons, stream width
16, norm form (eps = 0.1) and no-norm form, GELU (tanh form) and ReLU. Cuts: drop the k stream directions of least
energy (k = 2, 4, 8), projection reads, N's biases, re-set norm constants eps + c(D).
Run: uv run --with torch --with numpy --with sympy python model/tv_trained.py"""
import sys
import numpy as np
import torch

_argv = sys.argv
sys.argv = [_argv[0], "1"]          # tv_norm runs its own checks on import; 1 instance keeps that short
import tv_norm as T                 # noqa: E402
sys.argv = _argv

torch.manual_seed(0)
rng = np.random.default_rng(0)
DIN, d, L, q, C, NP = 8, 16, 3, 32, 4, 256
EPS = 0.1


def task(n):
    x = rng.normal(size=(n, DIN))
    s = np.stack([np.sin(2 * x[:, 0]) + x[:, 1] * x[:, 2], x[:, 3] ** 2 - x[:, 4], np.tanh(x[:, 5] * x[:, 6]),
                  0.5 * x[:, 7] - x[:, 0] * x[:, 4]], 1)
    return x.astype(np.float32), s.argmax(1)


def train(norm, kind, steps=3000):
    A = torch.nn.Linear(DIN, d)
    R = [torch.nn.Linear(d, q) for _ in range(L)]
    W = [torch.nn.Linear(q, d) for _ in range(L)]
    U = torch.nn.Linear(d, C)
    params = [p for m in [A, U] + R + W for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=3e-3)
    act = (lambda t: torch.nn.functional.gelu(t, approximate="tanh")) if kind == "gelu" else torch.relu
    nr = (lambda h: h / torch.sqrt((h * h).sum(-1, keepdim=True) + EPS)) if norm else (lambda h: h)

    def fwd(x):
        h = A(x)
        for l in range(L):
            h = h + W[l](act(R[l](nr(h))))
        return U(nr(h))
    xt, yt = task(4096)
    xt, yt = torch.tensor(xt), torch.tensor(yt)
    for _ in range(steps):
        i = torch.randint(0, 4096, (256,))
        loss = torch.nn.functional.cross_entropy(fwd(xt[i]), yt[i])
        opt.zero_grad(); loss.backward(); opt.step()
    xv, yv = task(2048)
    acc = float((fwd(torch.tensor(xv)).argmax(1).numpy() == yv).mean())
    g = lambda m: (m.weight.detach().numpy().astype(float), m.bias.detach().numpy().astype(float))
    xp, _ = task(NP)
    Aw, Ab = g(A)
    net = dict(act=kind, norm=norm, eps=[EPS] * (L + 1), e=xp.astype(float) @ Aw.T + Ab,
               R=[g(R[l])[0] for l in range(L)], b=[g(R[l])[1] for l in range(L)],
               W=[g(W[l])[0] for l in range(L)], c=[g(W[l])[1] for l in range(L)],
               U=g(U)[0], cy=g(U)[1])
    return net, acc


# point tv_norm's helpers at these sizes
T.d, T.L, T.q, T.C, T.NX = d, L, q, C, NP
T.Pperp = np.eye(C) - np.ones((C, C)) / C
rows = []
for norm in (True, False):
    for kind in ("gelu", "relu"):
        net, acc = train(norm, kind)
        hs, ns, As, yN = T.forward_N(net)
        Ecov = sum(h.T @ h for h in hs) / NP
        evals, evecs = np.linalg.eigh(Ecov)
        p = np.full(NP, 1 / NP)
        for k in (2, 4, 8):
            Dm = evecs[:, :k]
            PD = Dm @ Dm.T
            Pi = np.eye(d) - PD
            cD = [float(p @ ((hs[l] @ PD.T) ** 2).sum(1)) if norm else 0.0 for l in range(L + 1)]
            cut = dict(R=[R @ Pi for R in net["R"]], b=list(net["b"]), U=net["U"] @ Pi, cy=net["cy"],
                       eps=[EPS + cD[l] for l in range(L + 1)])
            hc, yB = T.forward_cut(net, cut, Pi)
            DP = T.kl(p, yN, yB)
            lk, om = T.leaks_omega(net, cut, Pi, hs, ns, yN)
            Lam, rem, _ = T.tv15(net, cut, Pi, hs, As, lk, om)
            true = (yB - yN) @ T.Pperp.T
            err = np.linalg.norm(true - Lam, axis=1)
            ok = bool((err <= rem * (1 + 1e-9) + 1e-12).all())
            b15 = 0.25 * (T.wnorm(p, np.linalg.norm(Lam, axis=1)) + T.wnorm(p, rem)) ** 2
            Lt, remt, _ = T.tv15(net, cut, Pi, hs, As, lk, om, tight=True)
            okt = bool((np.linalg.norm(true - Lt, axis=1) <= remt * (1 + 1e-9) + 1e-12).all())
            b15t = 0.25 * (T.wnorm(p, np.linalg.norm(Lt, axis=1)) + T.wnorm(p, remt)) ** 2
            b6 = T.tv6_bound(net, cut, Pi, [PD, Pi], hs, ns, p)
            lamonly = 0.25 * T.wnorm(p, np.linalg.norm(Lam, axis=1)) ** 2
            rows.append(f"| {'norm' if norm else 'no norm'} | {kind} | {acc:.2f} | {k} | {DP:.2e} | {b15 / DP:.3g} | "
                        f"{lamonly / DP:.3g} | {b15t / DP:.3g} | {okt} | {b6 / DP:.3g} | {T.wnorm(p, rem) / max(T.wnorm(p, np.linalg.norm(Lam, axis=1)), 1e-300):.2g} | {ok} |")
print("| form | act | test acc | k dropped | D_P | TV15 / D_P | (Lambda only) / D_P | TV15 tight / D_P | tight ok | TV6 / D_P | ||rem|| / ||Lambda|| | pointwise ok |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
print("\n".join(rows))
