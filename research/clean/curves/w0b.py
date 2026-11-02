"""Card W0b (microscope): where does the second-order picture break?
(a) scale: real KL of intact + t * du against t^2 * predicted, t in {.1, .25, .5, 1}.
(b) expansion point: the same quadratic prediction, but with the Fisher taken after 4a (exact change through the 4a
    ReLUs), after 4b, after 4c. If one level matches the real KL, the nonlinearity before it is what breaks W0."""
import numpy as np, torch
from inception import *
from family import Data, kl, IN4A
from d4_write import stats, writes
from w_fisher import intact, upper, A_cols, du, coarse, load_build, fisher_grads, PAD


def to_4a(m, outs):
    a = m.mixed4a_1x1(outs[0]); b = m.mixed4a_3x3_bottleneck(outs[1]); cc = m.mixed4a_5x5_bottleneck(outs[2])
    d = m.mixed4a_pool_reduce(outs[3])
    b = m.mixed4a_3x3(m.mixed4a_3x3_pre_relu_conv(F.pad(b, (1, 1, 1, 1))))
    cc = m.mixed4a_5x5(m.mixed4a_5x5_pre_relu_conv(F.pad(cc, (2, 2, 2, 2))))
    return torch.cat([a, b, cc, d], 1)


def after(m, h, start):
    """From the output of block `start` (4a, 4b, 4c) to logits."""
    names = ["mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed4e"]
    for n in names[names.index(start) + 1:]:
        h = mixed(m, n, h)
    h = m.maxpool10(F.pad(h, (0, 1, 0, 1), value=float('-inf')), kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
    for n in ["mixed5a", "mixed5b"]:
        h = mixed(m, n, h)
    return m.softmax2_pre_activation_matmul(F.avg_pool2d(h, 7).reshape(-1, 1024))


def grads_at(m, h, start, R=10):
    h = h.detach().requires_grad_(); lp = F.log_softmax(after(m, h, start), 1); top = lp.detach().exp().topk(R, 1); G = []
    for r in range(R):
        G.append(torch.autograd.grad((top.values[:, r].sqrt() * lp.gather(1, top.indices[:, r:r + 1])[:, 0]).sum(), h, retain_graph=r < R - 1)[0])
    return G


if __name__ == "__main__":
    m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"])
    for p in m.parameters(): p.requires_grad_(False)
    fd = load_build(m, fam, "derived2-r1-ks7"); ft = load_build(m, fam, "L-k1-r-lr3e-2")
    res, _ = writes(m, fam, stats(m, fd, fam, D)); fw = torch.load("cache/fisher_writes.pt")
    tw = ft.weights()[2]
    builds = {"der+bayes": (fd, [c.to(DEV) for c in res["bayes"][0]], res["bayes"][1]),
              "der+vmr": (fd, [c.to(DEV) for c in res["vmr"][0]], res["vmr"][1]),
              "der+fisher": (fd, [c.to(DEV) for c in fw["der+fisher"][0]], [b.to(DEV) for b in fw["der+fisher"][1]]),
              "trn+trained": (ft, tw, [torch.zeros(c.shape[0], device=DEV) for c in tw])}
    A = A_cols(m, fam); idx = np.sort(np.random.default_rng(5).choice(D.dv, 400, replace=False)); T = [.1, .25, .5, 1.0]
    acc = {k: dict(real={t: [] for t in T}, p_in=[], p4a=[], p4b=[], p4c=[]) for k in builds}
    for i in range(0, len(idx), 20):
        b = idx[i:i + 20]
        with torch.no_grad():
            ins = inputs_3b(m, to_3a(m, prep(D.X[b]))); outs, z1, z2 = intact(m, fam, ins)
            h4a = to_4a(m, outs); h4b = mixed(m, "mixed4b", h4a); h4c = mixed(m, "mixed4c", h4b); t0 = after(m, h4c, "mixed4c")
        Gin, _ = fisher_grads(m, outs, 10); G4a = grads_at(m, h4a, "mixed4a"); G4b = grads_at(m, h4b, "mixed4b"); G4c = grads_at(m, h4c, "mixed4c")
        with torch.no_grad():
            for k, (fm, cols, cb) in builds.items():
                c1, c2 = coarse(fm, ins); d = du(A, z1, z2, cols, cb, c1, c2)
                for t in T:
                    acc[k]["real"][t].append(kl(t0, upper(m, [o - t * dj for o, dj in zip(outs, d)])).cpu().numpy())
                acc[k]["p_in"].append((sum(sum((g[j] * d[j]).sum((1, 2, 3)) for j in range(4)) ** 2 for g in Gin) / 2).cpu().numpy())
                c4a = to_4a(m, [o - dj for o, dj in zip(outs, d)]); c4b = mixed(m, "mixed4b", c4a); c4c = mixed(m, "mixed4c", c4b)
                for key, G, dh in [("p4a", G4a, c4a - h4a), ("p4b", G4b, c4b - h4b), ("p4c", G4c, c4c - h4c)]:
                    acc[k][key].append((sum((g * dh).sum((1, 2, 3)) ** 2 for g in G) / 2).cpu().numpy())
    for k, a in acc.items():
        real = np.concatenate(a["real"][1.0]); pin = np.concatenate(a["p_in"])
        sc = " ".join(f"t={t}: {np.concatenate(a['real'][t]).mean() / (t * t * pin.mean()):.2f}" for t in T)
        print(f"{k:11s} real {real.mean():.4f} | real/(t^2 pred_in): {sc}", flush=True)
        for key in ["p_in", "p4a", "p4b", "p4c"]:
            p = np.concatenate(a[key]); print(f"     pred at {key:4s}: {p.mean():.4f}  ratio real/pred {real.mean() / p.mean():.2f}  per-image corr {np.corrcoef(p, real)[0, 1]:.2f}", flush=True)
