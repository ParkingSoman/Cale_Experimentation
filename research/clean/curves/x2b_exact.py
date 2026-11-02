"""R2b (loop step 8, NEURON-MATH section 14): which reason separates level 1 from the KL? Evaluate L2, L3, L3' and KL
for six builds on 500 dev images outside dev-1000 (none was tuned on them). Labels y drawn from the intact network."""
import numpy as np, torch
from inception import *
import b3_train
from family import Data, kl, pool3, IN4A
m = load_model(); D = Data(seed=0); D1 = Data(seed=0, ndev=1000); fam = list(np.load("cache/s1_fine.npz")["fam"])
idx = np.sort(np.random.default_rng(3).choice(np.setdiff1d(D.dv, D1.dv), 500, replace=False))
convs = [getattr(m, n) for n in IN4A]; NY = 8
def cont(post):                       # rest of the network from the four post-ReLU reader outputs
    a, b, cc, d = post
    b = m.mixed4a_3x3(m.mixed4a_3x3_pre_relu_conv(F.pad(b, (1, 1, 1, 1))))
    cc = m.mixed4a_5x5(m.mixed4a_5x5_pre_relu_conv(F.pad(cc, (2, 2, 2, 2))))
    hh = torch.cat([a, b, cc, d], 1)
    for n in ["mixed4b", "mixed4c", "mixed4d", "mixed4e"]: hh = mixed(m, n, hh)
    hh = pool3(hh)
    for n in ["mixed5a", "mixed5b"]: hh = mixed(m, n, hh)
    return m.softmax2_pre_activation_matmul(F.avg_pool2d(hh, 7).reshape(-1, 1024))
def reader_pre(x, p, fm=None):
    out = []
    for j, (c, inp) in enumerate(zip(convs, [x, x, x, p])):
        if fm is None: out.append(F.conv2d(inp, c.weight, c.bias)); continue
        keep_in = inp[:, fm.keep]; W = (c.weight[:, fm.keep, 0, 0] + fm.dW[j] * fm.cscale[j])[:, :, None, None]
        out.append(F.conv2d(keep_in, W, c.bias + fm.colbias[j]))
    return out
names = {"removal": None, "level0": "x2_a_level0_x", "level0p": "x2_b_level0_p", "level1": "x2_c_level1_n2000",
         "trained_zero_long": "B3-k0-lr003-long", "trained_from_level0": "B3-k0-formulastart"}
builds = {}
for k, v in names.items():
    fm = b3_train.FamilyRA(m, fam, 1); fm.add_rb(m); fm.add_ra(m)
    with torch.no_grad():
        for cc in fm.cols: cc.zero_()
    if v and v.startswith("x2_"): fm.load_state_dict(torch.load(f"cache/{v}.pt", map_location="cpu"))
    elif v: fm = b3_train.load_b3(m, v).cpu()
    builds[k] = fm.to(DEV).eval()
acc = {k: dict(KL=[], L2=[], L3=[], L3p=[]) for k in builds}
g = torch.Generator(device="cpu").manual_seed(0)
for i in range(0, len(idx), 20):
    b = idx[i:i + 20]
    with torch.no_grad():
        x = pool3(F.relu(pre_3b(m, inputs_3b(m, to_3a(m, prep(D.X[b])))[0])))
        p = F.max_pool2d(F.pad(x, (1, 1, 1, 1), value=float('-inf')), 3, 1)
        a0 = reader_pre(x, p); T = torch.from_numpy(D.T[b].astype(np.float32)).to(DEV)
        E, Dl = {}, {}
        for k, fm in builds.items():
            a1 = reader_pre(x, p, fm); E[k] = [u - v for u, v in zip(a0, a1)]
            Dl[k] = [F.relu(u) - F.relu(v) for u, v in zip(a0, a1)]
            acc[k]["KL"].append(kl(T, cont([F.relu(v) for v in a1])).cpu().numpy())
    pre = [t.clone().requires_grad_(True) for t in a0]
    post = [F.relu(t) for t in pre]; [t.retain_grad() for t in post]
    lp = F.log_softmax(cont(post), 1)
    ys = torch.multinomial(lp.detach().exp().cpu(), NY, replacement=True, generator=g).to(DEV)
    f2 = [torch.zeros_like(t) for t in pre]; s3 = {k: torch.zeros(len(b), NY, device=DEV) for k in builds}
    s3p = {k: torch.zeros(len(b), NY, device=DEV) for k in builds}
    for s in range(NY):
        for t in pre + post: t.grad = None
        lp.gather(1, ys[:, s:s + 1]).sum().backward(retain_graph=s < NY - 1)
        with torch.no_grad():
            for j in range(4): f2[j] += pre[j].grad ** 2 / NY
            for k in builds:
                s3[k][:, s] = sum((pre[j].grad * E[k][j]).flatten(1).sum(1) for j in range(4))
                s3p[k][:, s] = sum((post[j].grad * Dl[k][j]).flatten(1).sum(1) for j in range(4))
    with torch.no_grad():
        for k in builds:
            acc[k]["L2"].append((0.5 * sum((f2[j] * E[k][j] ** 2).flatten(1).sum(1) for j in range(4))).cpu().numpy())
            acc[k]["L3"].append((0.5 * (s3[k] ** 2).mean(1)).cpu().numpy())
            acc[k]["L3p"].append((0.5 * (s3p[k] ** 2).mean(1)).cpu().numpy())
    with torch.no_grad(): pr = lp.detach().exp()
    var = lambda t: (pr * t ** 2).sum(1) - (pr * t).sum(1) ** 2
    for k in builds:
        with torch.no_grad():   # J e by central difference (network above the readers is piecewise linear)
            fdp = lambda eps: (cont([F.relu(t + eps * u) for t, u in zip(a0, E[k])]) - cont([F.relu(t - eps * u) for t, u in zip(a0, E[k])])) / (2 * eps)
            fdq = lambda eps: (cont([F.relu(t) + eps * u for t, u in zip(a0, Dl[k])]) - cont([F.relu(t) - eps * u for t, u in zip(a0, Dl[k])])) / (2 * eps)
            jp, jq = fdp(1e-2), fdq(1e-2)
            if i == 0: print(k, "FD check eps 1e-2 vs 1e-3: rel diff", float((var(jp) - var(fdp(1e-3))).abs().sum() / var(jp).abs().sum()), flush=True)
        acc[k]["L3"][-1] = (0.5 * var(jp)).detach().cpu().numpy(); acc[k]["L3p"][-1] = (0.5 * var(jq)).detach().cpu().numpy()
res = {k: {q: np.concatenate(v) for q, v in d.items()} for k, d in acc.items()}
np.savez("cache/x2b_reason_exact.npz", **{f"{k}_{q}": v for k, d in res.items() for q, v in d.items()})
print(f"{'build':22s} {'KL':>8s} {'L2':>8s} {'L3':>8s} {'L3p':>8s}")
for k, d in res.items():
    print(f"{k:22s} " + " ".join(f"{d[q].mean():8.4f}" for q in ["KL", "L2", "L3", "L3p"]))
ref = res["level1"]
print("paired differences, trained_from_level0 minus level1 (se):")
for q in ["KL", "L2", "L3", "L3p"]:
    dd = res["trained_from_level0"][q] - ref[q]; print(f"  {q:4s} {dd.mean():+.4f} ({dd.std() / np.sqrt(len(dd)):.4f})  ratio {res['trained_from_level0'][q].mean() / ref[q].mean():.3f}")
print("per-image corr with KL across all builds:", {q: round(float(np.corrcoef(np.concatenate([res[k]['KL'] for k in res]), np.concatenate([res[k][q] for k in res]))[0, 1]), 3) for q in ["L2", "L3", "L3p"]})
