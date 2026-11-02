"""Part VII section 26: gated transport of the output metric to the family's 3920 signals.
Masked LINEAR copy of the network above the readers (biases off; J at reader outputs). Gates (every ReLU from the 4a
3x3 / 5x5 outputs to 5b) and max-pool selections come from:
  true  = the real network on each image (exact Jacobian of that image)
  bern  = independent Bernoulli(r) per unit and position (r = image firing rates), uniform one-hot pool selection
Output metric: 'img' = each image's own top-10 class covariance (as Kbar), 'avg' = its image average, 'uni' = I - 11^T/n.
Variants: a1 = true + img (must equal Kbar: code check), a2 = true + avg, a3 = true + uni, b = bern + avg, b_uni = bern + uni.
Also the per-depth weights of Prop. 26.1 for the Bernoulli model. Each K gets the factorised build."""
import numpy as np, torch, time, sys, copy
from circ import *
t0 = time.time(); VAR = sys.argv[1].split(",") if len(sys.argv) > 1 else ["a1", "a2", "a3", "b", "b_uni"]
mlin = copy.deepcopy(m)
for _, mm in mlin.named_modules():
    if isinstance(mm, (torch.nn.Conv2d, torch.nn.Linear)) and mm.bias is not None: mm.bias.data.zero_()
BLOCKS = ["mixed4b", "mixed4c", "mixed4d", "mixed4e", "mixed5a", "mixed5b"]
def pad_ninf(t, pd): return F.pad(t, pd, value=float('-inf'))

def net(net_m, stage0, G, P):
    """stage0: the four reader outputs (list). G(name, pre) -> gated tensor; P(name, h, kind) -> pooled tensor."""
    a, b, c, d = stage0
    b = G("4a_3x3", net_m.mixed4a_3x3_pre_relu_conv(F.pad(b, (1, 1, 1, 1))))
    c = G("4a_5x5", net_m.mixed4a_5x5_pre_relu_conv(F.pad(c, (2, 2, 2, 2))))
    h = torch.cat([a, b, c, d], 1)
    for n in BLOCKS:
        if n == "mixed5a": h = P("pool4e", h, 2)
        g = lambda s_: getattr(net_m, f"{n}_{s_}")
        a = G(n + "_1x1", g("1x1_pre_relu_conv")(h))
        b = G(n + "_3x3", g("3x3_pre_relu_conv")(F.pad(G(n + "_3x3b", g("3x3_bottleneck_pre_relu_conv")(h)), (1, 1, 1, 1))))
        c = G(n + "_5x5", g("5x5_pre_relu_conv")(F.pad(G(n + "_5x5b", g("5x5_bottleneck_pre_relu_conv")(h)), (2, 2, 2, 2))))
        d = G(n + "_pr", g("pool_reduce_pre_relu_conv")(P(n + "_pool", h, 1)))
        h = torch.cat([a, b, c, d], 1)
    return net_m.softmax2_pre_activation_matmul(F.avg_pool2d(h, 7).reshape(-1, 1024))

def windows(h, stride):
    hp = F.pad(h, (1, 1, 1, 1)) if stride == 1 else F.pad(h, (0, 1, 0, 1))
    U = F.unfold(hp, 3, stride=stride); B, C = h.shape[:2]; return U.reshape(B, C, 9, -1), (h.shape[2] if stride == 1 else 7)
def valid(h, stride):
    one = torch.ones_like(h[:1, :1]); U, _ = windows(one, stride); return U[0, 0] > 0          # (9, L) window positions inside the map

def record(x, p):
    """real network on images: gates and pool argmax (window index 0..8) for each layer."""
    rec = {}
    def G(name, pre): rec[name] = pre > 0; return F.relu(pre)
    def P(name, h, stride):
        hp = pad_ninf(h, (1, 1, 1, 1) if stride == 1 else (0, 1, 0, 1)); out, ind = F.max_pool2d(hp, 3, stride, return_indices=True)
        s = out.shape[2]; Wp = hp.shape[3]; L = torch.arange(s * s, device=DEV); oy, ox = (L // s) * stride, (L % s) * stride
        ind = ind.flatten(2); rec[name] = (ind // Wp - oy) * 3 + (ind % Wp - ox)       # torch's own choice (matters for ties at 0)
        return out
    with torch.no_grad(): lg = net(m, readers_post(x, p), G, P)
    return rec, lg

def lin_true(rec, rep):
    def G(name, pre): return pre * rec[name].repeat_interleave(rep, 0)
    def P(name, h, stride):
        U, s = windows(h, stride); idx = rec[name].repeat_interleave(rep, 0)
        return U.gather(2, idx[:, :, None]).squeeze(2).reshape(h.shape[0], h.shape[1], s, s)
    return G, P

def lin_bern(rates):
    def G(name, pre): return pre * (torch.rand_like(pre) < rates[name]).float()
    def P(name, h, stride):
        U, s = windows(h, stride); v = valid(h, stride)
        sc = torch.rand(U.shape, device=DEV).masked_fill(~v[None, None], -1.0); idx = sc.argmax(2)
        return U.gather(2, idx[:, :, None]).squeeze(2).reshape(h.shape[0], h.shape[1], s, s)
    return G, P

def fam_grads(G, P, xi):
    """g = J^T xi for each row of xi (B, 1008), J = masked linear map from the family's 3920 signals to the logits."""
    s = torch.zeros(xi.shape[0], 2, 10, 14, 14, device=DEV, requires_grad=True)
    x, p = s[:, 0], s[:, 1]
    stage0 = [F.conv2d(inp, WF[j][:, :, None, None]) for j, inp in enumerate([x, x, x, p])]
    lg = net(mlin, stage0, G, P); (g,) = torch.autograd.grad((lg * xi).sum(), s)
    return g.reshape(xi.shape[0], 3920)

# pass 1: rates, average output covariance, and (for a1) per-image top-10 weights
rates = {}; Savg = torch.zeros(1008, 1008, device=DEV); n_r = 0
for i in range(0, 2000, 20):
    x, p = family_inputs(TR[i:i + 20]); rec, lg = record(x, p)
    for k, v in rec.items():
        if v.dtype == torch.bool: rates[k] = rates.get(k, 0) + v.float().sum(0)
    pr = lg.softmax(1); top = pr.topk(10, 1); pk = top.values / top.values.sum(1, keepdim=True)
    E = torch.zeros(len(pr), 10, 1008, device=DEV).scatter_(2, top.indices[:, :, None], 1.0)
    Ec = (E - (pk[:, :, None] * E).sum(1, keepdim=True)) * pk[:, :, None].sqrt(); Savg += torch.einsum("bci,bcj->ij", Ec, Ec); n_r += len(pr)
rates = {k: v / n_r for k, v in rates.items()}; Savg /= n_r
ev, evec = torch.linalg.eigh(Savg.cpu().double()); Sroot = (evec * ev.clamp(min=0).sqrt()) @ evec.T; Sroot = Sroot.float().to(DEV)
print("rates:", {k: round(float(v.mean()), 2) for k, v in list(rates.items())[:8]}, "...", f"({time.time() - t0:.0f}s)", flush=True)
torch.save({k: v.cpu() for k, v in rates.items()}, "cache/c20_rates.pt")
uni = lambda B: (lambda z: z - z.mean(1, keepdim=True))(torch.randn(B, 1008, device=DEV))

def K_true(metric):
    K = torch.zeros(3920, 3920, device=DEV)
    for i in range(0, 2000, 10):
        x, p = family_inputs(TR[i:i + 10]); rec, lg = record(x, p); B = len(lg)
        if metric == "img":
            pr = lg.softmax(1); top = pr.topk(10, 1); pk = top.values / top.values.sum(1, keepdim=True)
            E = torch.zeros(B, 10, 1008, device=DEV).scatter_(2, top.indices[:, :, None], 1.0)
            xi = ((E - (pk[:, :, None] * E).sum(1, keepdim=True)) * pk[:, :, None].sqrt()).reshape(B * 10, 1008)
        elif metric == "avg": xi = torch.randn(B * 10, 1008, device=DEV) @ Sroot
        else: xi = uni(B * 10)
        G, P = lin_true(rec, 10); g = fam_grads(G, P, xi); K += g.T @ g
    return (K / 2000).cpu().double()

def K_bern(metric, draws=2000, per=10):
    K = torch.zeros(3920, 3920, device=DEV)
    for i in range(0, draws, 10):
        B = 10 * per; xi = torch.randn(B, 1008, device=DEV) @ Sroot if metric == "avg" else uni(B)
        G, P = lin_bern(rates)
        # one mask draw per row: masks are drawn inside G / P with the batch shape, so every row has its own gates
        g = fam_grads(G, P, xi); K += g.T @ g
    return (K / draws).cpu().double()

Ks = {}
for v in VAR:
    if v == "a1": Ks[v] = K_true("img"); Kbar = torch.load("cache/x5_kbar.pt").double(); print(f"a1 code check: |K - Kbar| / |Kbar| = {float((Ks[v] - Kbar).norm() / Kbar.norm()):.2e}", flush=True)
    elif v == "a2": Ks[v] = K_true("avg")
    elif v == "a3": Ks[v] = K_true("uni")
    elif v == "b": Ks[v] = K_bern("avg")
    elif v == "b_uni": Ks[v] = K_bern("uni")
    torch.save(Ks[v].float(), f"cache/c20_K_{v}.pt")
    e = torch.linalg.eigvalsh(Ks[v]).clamp(min=0); pe = e / e.sum(); erank = float(torch.exp(-(pe[pe > 0] * pe[pe > 0].log()).sum()))
    print(f"[{v}] K built, effective rank {erank:.0f}  ({time.time() - t0:.0f}s)", flush=True)
refs = REFS(); refs["4d"] = np.load("cache/c8_ladder_4b_4d_5b_logits.npz")["4d"]; res = {}
for v in VAR:
    res[v] = factor_build(Ks[v], f"c20_build_{v}"); paired(v, res[v], refs)
np.savez(f"cache/c20_depth_{'_'.join(VAR)}.npz", **res); print(f"done ({time.time() - t0:.0f}s)")
