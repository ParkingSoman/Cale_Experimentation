"""T1-T6: free-coefficient reconstruction (Theorem 23) on the toy. usage: t_kkt.py [steps] [all|fc]"""
import sys, time, torch, torch.nn.functional as F
from shapes import DEV, LAM, Net, sample, residual
from gens import g3_kkt, g4_varpro
from proxy import unc, cofiring, fit, score
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000; which = sys.argv[2] if len(sys.argv) > 2 else "all"
ck = torch.load("net_1000.pt"); net = Net().to(DEV); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
X, y = ck["X"].to(DEV), ck["y"].to(DEV); N = len(X); Xtruth, _ = sample(1000, 7); Xeval, _ = sample(5000, 8)
# T1 instrument: the true train set with c = p_k / (N lam) in the same residual
ps = list(net.parameters()); [p.requires_grad_(True) for p in ps]
f = net(X); c = (F.softmax(f, 1) / (N * LAM)).detach() * (1 - F.one_hot(y, 4).float())
g = torch.autograd.grad((c * (f.gather(1, y[:, None]) - f)).sum(), ps); th = [p.detach() for p in ps]
print(f"T1 floor: residual with the true train set {float(torch.sqrt(sum(((t - a) ** 2).sum() for a, t in zip(g, th))) / torch.sqrt(sum((t ** 2).sum() for t in th))):.4f}")
[p.requires_grad_(False) for p in ps]
layers = ["fc.weight", "fc.bias"] if which == "fc" else None
t0 = time.time()
if which == "varpro": xs, ys, cs, hist = g4_varpro(net, 1000, steps=steps, K=300, log=print)
else: xs, ys, cs, hist = g3_kkt(net, 1000, steps=steps, layers=layers, log=print)
print(f"generator ({which}) {time.time() - t0:.0f} s")
print(f"T2: final residual {hist[-1]:.4f}, min {min(hist):.4f}; steps with residual in [.98, 1.02]: {sum(.98 <= h <= 1.02 for h in hist)} of {len(hist)}; sum c {float(cs.sum()):.3f} (true-set sum {float(c.sum()):.3f})")
# T3: nearest training image, and how unsure those training images are
Z = lambda A: (A.flatten(1) - A.flatten(1).mean(1, keepdim=True)) / A.flatten(1).std(1, keepdim=True).clamp(min=1e-6)
corr = Z(xs) @ Z(X).T / X[0].numel(); best, nn = corr.max(1); hit = best > .8; ut = unc(net, X)
print(f"T3: share with a training image at corr > .8: {float(hit.float().mean()):.3f} (> .6: {float((best > .6).float().mean()):.3f}); median best {float(best.median()):.3f}")
if hit.any(): print(f"    1 - sum p^2 of the matched training images {float(ut[nn[hit]].mean()):.4f} vs all training images {float(ut.mean()):.4f}; distinct matched {len(set(nn[hit].tolist()))}")
# T4: self-consistency of c with the network's p on the candidates (pairs of wrong classes)
p = F.softmax(net(xs), 1); lc, lp = [], []
for s in range(len(xs)):
    k = [j for j in range(4) if j != int(ys[s])]
    for i1 in range(3):
        for i2 in range(i1 + 1, 3):
            a, b = k[i1], k[i2]
            if cs[s, a] > 1e-6 and cs[s, b] > 1e-6: lc.append(float(torch.log(cs[s, a] / cs[s, b]))); lp.append(float(torch.log(p[s, a] / p[s, b])))
print(f"T4: corr(log c ratio, log p ratio) = {float(torch.corrcoef(torch.tensor([lc, lp]))[0, 1]) if len(lc) > 2 else float('nan'):.3f} over {len(lc)} pairs")
# T5: P2 proxy; references
sets = {"truth (fresh)": Xtruth, "train set": X, "train unsure (u > .1)": X[ut > .1], f"G3 rebuilt ({which})": xs}
print(f"train unsure count: {int((ut > .1).sum())}; G3 candidates 1 - sum p^2 {float(unc(net, xs).mean()):.4f}; co-firing vs truth {cofiring(net, xs, Xtruth):+.3f}")
ref = {fi: score(net, fit(net, Xtruth, fi), Xeval) for fi in [False, True]}
for k, S in sets.items():
    for fi in [False, True]:
        v = score(net, fit(net, S, fi), Xeval); d = v - ref[fi]
        print(f"T5 {'Fisher' if fi else 'lsq   '} write from {k:24s}: KL {float(v.mean()):.4f} (se {float(v.std() / 5000 ** .5):.4f}) | paired vs truth {float(d.mean()):+.4f} (se {float(d.std() / 5000 ** .5):.4f})")
torch.save(dict(xs=xs.cpu(), ys=ys.cpu(), cs=cs.cpu(), hist=hist), f"g3_{which}_{steps}.pt")
