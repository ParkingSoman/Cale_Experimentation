"""P5 Rank (FANOUT round 1, builder step): read-aware greedy ranking. ONE nested order gives every size. Plain cut.

Part changed: P5 Rank only (which units stay). No weight is changed. Writes, gates and stream maps stay the floor's.
Derived from: THEORY 1 (written x read, weakest first, one ranking), THEORY 2 (activation x output Jacobian),
BOARD F1 (activity must enter the score), the P5 revised proposal in round1/talk.json (turn2) and the critic's fixes
(ln_pre masked; null = same formula with a permuted pairing inside each set).
Photo use: NONE. The ranking uses weights plus the generic input X (board.py --input), or the default pink input.

MATH
  Every unit u has a mask m[n,u] per input n (a leaf tensor). Masked unit output = m * a.
  Probe: CLIP o_n = 100 e_n / ||e_n||, J_n = r_n . o_n, r_n ~ N(0, I_512).
         ResNet J_n = r_n . z_n (logits), r_n ~ N(0, I - 11^T/1000) (centred Gaussian).
         r_n is drawn once from seed (row n = input n), so every greedy step and the null use the same r_n.
  G[n,u] = dJ_n / dm[n,u]   (= sum over tokens of a * g for a head, MLP neuron, ResNet inner or stream channel)
  Score S_u = (1/N) sum_n G[n,u]^2   (sum over tokens, square per input, then average).
  Greedy schedule SCHED: score the kept units on the MASKED network, then in each set mask the lowest kept units
  until the set has the next size (vitgen.sizes / wide18.widths). The cache keeps, per unit, the step at removal
  and the score at removal (survivors: the last score). Keep(k) = the k units removed last. So 95% contains 50%,
  which contains 25%, which contains 5%.

MASK POINTS
  CLIP: stream dim = ONE mask (N, 768) used at every stream read: the inputs of ln_pre, ln1 and ln2 of all blocks,
        and ln_post. Masked LN over the kept dims: mu = sum(m x)/sum(m), var = sum(m (x-mu)^2)/sum(m),
        y = m * ((x-mu)/sqrt(var+eps) g + b). Divisor = the kept count (norm mode "kept" = today's vitgen.cut).
        ln_pre masked, so cut dims start at 0 and only collect writes that nothing reads. Residual adds unmasked.
        Head h of block i: mask on its 64-d output before o_w. MLP neuron j: mask on the GELU output.
  ResNet: inner channel of block (s,b): mask on relu(bn1(conv1 .)). Stream channel of stage s: one mask at every
        READ of that stream (conv1 input of its blocks, conv1 and downsample input of the next stage's block 0,
        avgpool input for stage 4). Identity skips unmasked. BN = the fixed eval affine map (no statistics as data).
  With binary masks the masked forward equals the plain cut (checked by distill/fanout/p5check.py).

NULL (rules/p5rank_null.py): same formula, read pairing permuted inside each set:
  G_null[n,u] = sum_t a[n,u,t] g[n,pi(u),t], pi a fixed single-cycle permutation inside the set (seed 1000+seed).
  CLIP stream set: the masked LN adds count and centring terms that are not a plain a*g. Definition used here:
  G_null = dJ/dm - sum_sites sum_t z_u gy_u + sum_sites sum_t z_u gy_pi(u), with z the LN output before the mask
  and gy = dJ/dy. Only the direct read pairing is permuted; the norm-statistics terms stay the rule's.
  The CLIP null order stops at f = .5 (NULL_STOP) to keep the Mac time short; asking for f < .5 raises.

API for other parts: order(net, model, X, seed, null, stop) -> cache dict; keep(C, name, k) -> sorted kept indices;
rank(C, name) -> int rank per unit (0 = removed first); score(net, model, X, seed) -> step-1 scores (cached).
Cache: fanout/cache/p5rank_{net}_{input tag}_s{seed}[_null][_to{stop}]_v{VERSION}.pt. Input tag = hash of X.
"""
import copy
import hashlib
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
FAN = os.path.dirname(HERE)
sys.path.insert(0, FAN)
sys.path.insert(0, os.path.dirname(FAN))
import generic  # noqa: E402
import vitgen  # noqa: E402

VERSION = 1
CACHE = os.path.join(FAN, "cache")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
SCHED = (1.0, .95, .85, .75, .65, .5, .4, .3, .25, .18, .12, .08, .05)
BS = {"clip": 16, "r18": 64}
N_READ = {"clip": 32, "r18": 256}       # inputs used for the per-part probe-error readout
NULL_STOP = {"clip": 0.5, "r18": None}
_MEM = {}


def log(*a):
    print("[p5rank]", *a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- input, probe, sets
def default_input():
    """The same pink input noisecut draws by default (256 images, batch seeds 100000 + a, batches of 32)."""
    return torch.cat([generic.pink(32, seed=100000 + a) for a in range(0, 256, 32)])


def input_tag(X):
    if X is None:
        return "pinkdefault"
    h = hashlib.sha1(X.detach().cpu().to(torch.float16).contiguous().numpy().tobytes()).hexdigest()[:10]
    return f"n{len(X)}_{h}"


def probe(net, N, seed, dim):
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(N, dim, generator=g)
    return r if net == "clip" else r - r.mean(1, keepdim=True)


def sizes_fn(net, model):
    """f -> {set name: kept count}."""
    if net == "clip":
        nb = len(model["blocks"])

        def fn(f):
            d, h, m = vitgen.sizes(f, model)
            out = {"stream": d}
            out.update({f"head{i}": h for i in range(nb)})
            out.update({f"mlp{i}": m for i in range(nb)})
            return out
        return fn
    import wide18

    def fn(f):
        w = wide18.widths(f)
        out = {f"stream{s}": w[s] for s in range(4)}
        out.update({f"inner{s}{b}": w[s] for s in range(4) for b in range(2)})
        return out
    return fn


def parts(net, names):
    if net == "clip":
        return {"stream": ["stream"], "heads": [k for k in names if k.startswith("head")],
                "mlp": [k for k in names if k.startswith("mlp")]}
    return {"stream": [k for k in names if k.startswith("stream")], "inner": [k for k in names if k.startswith("inner")]}


def make_perm(sets, seed):
    """One fixed single-cycle permutation per set (no unit is paired with itself when n > 1)."""
    g = torch.Generator().manual_seed(1000 + seed)
    out = {}
    for k in sorted(sets):
        n = sets[k]
        p = torch.randperm(n, generator=g)
        pi = torch.empty(n, dtype=torch.long)
        pi[p] = p.roll(-1)
        out[k] = pi
    return out


# ---------------------------------------------------------------- null recorder (hooks)
class Rec:
    """Records sum_t a[u] * g[pi(u)] for each unit set during backward (the null's pairing)."""

    def __init__(self, perm):
        self.perm = perm
        self.G = {}

    def _acc(self, k, v):
        self.G[k] = v if k not in self.G else self.G[k] + v

    def unit(self, name, a, am, axis, sdims):
        p, a = self.perm[name], a.detach()

        def hook(g, a=a, p=p, name=name, axis=axis, sdims=sdims):
            self._acc(name, (a * g.index_select(axis, p)).sum(sdims))
        am.register_hook(hook)

    def stream_site(self, z, y):
        p, z = self.perm["stream"], z.detach()

        def hook(g, z=z, p=p):
            d, dp = z * g, z * g.index_select(-1, p)
            if z.dim() == 3:
                d, dp = d.sum(1), dp.sum(1)
            self._acc("_D", d)
            self._acc("_Dp", dp)
        y.register_hook(hook)


# ---------------------------------------------------------------- CLIP masked forward
def _mln(x, n, m, rec=None):
    assert n["mode"] == "ln", "p5rank: norm mode 'kept' only (rho divisor not built yet)"
    mm = m[:, None, :] if x.dim() == 3 else m
    cnt = mm.sum(-1, keepdim=True)
    mu = (mm * x).sum(-1, keepdim=True) / cnt
    xc = x - mu
    var = (mm * xc * xc).sum(-1, keepdim=True) / cnt
    z = xc * torch.rsqrt(var + vitgen.EPS) * n["g"] + n["b"]
    y = mm * z
    if rec is not None:
        rec.stream_site(z, y)
    return y


def vit_fwd(Td, x, M, rec=None):
    """Masked CLIP vision tower. M: {'stream': (N, d), 'head{i}': (N, H), 'mlp{i}': (N, m)}. Returns e (N, 512)."""
    t = F.conv2d(x, Td["patch_w"], stride=16).flatten(2).transpose(1, 2)
    t = torch.cat([Td["cls"].expand(len(x), 1, -1), t], 1) + Td["pos"]
    ms = M["stream"]
    s = _mln(t, Td["ln_pre"], ms, rec)
    for i, B in enumerate(Td["blocks"]):
        N, L, _ = s.shape
        H = B["q_w"].shape[0] // vitgen.HEAD_DIM
        y = _mln(s, B["ln1"], ms, rec)
        q = F.linear(y, B["q_w"], B["q_b"]).view(N, L, H, vitgen.HEAD_DIM).transpose(1, 2)
        k = F.linear(y, B["k_w"], B["k_b"]).view(N, L, H, vitgen.HEAD_DIM).transpose(1, 2)
        v = F.linear(y, B["v_w"], B["v_b"]).view(N, L, H, vitgen.HEAD_DIM).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)                       # (N, H, L, 64)
        om = o * M[f"head{i}"][:, :, None, None]
        if rec is not None:
            rec.unit(f"head{i}", o, om, 1, (2, 3))
        s = s + F.linear(om.transpose(1, 2).reshape(N, L, H * vitgen.HEAD_DIM), B["o_w"], B["o_b"])
        y = _mln(s, B["ln2"], ms, rec)
        g = F.gelu(F.linear(y, B["fc_w"], B["fc_b"]))                     # (N, L, m)
        gm = g * M[f"mlp{i}"][:, None, :]
        if rec is not None:
            rec.unit(f"mlp{i}", g, gm, 2, (1,))
        s = s + F.linear(gm, B["pr_w"], B["pr_b"])
    return _mln(s[:, 0], Td["ln_post"], ms, rec) @ Td["proj"]


def clip_out(Td, x, M, rec=None):
    return 100 * F.normalize(vit_fwd(Td, x, M, rec), dim=-1)


# ---------------------------------------------------------------- ResNet masked forward
def prep_r18(A, dev=DEV, dtype=torch.float32):
    """A private copy on dev, eval, no parameter gradients (A itself is not touched)."""
    Am = copy.deepcopy(A).to(dev).to(dtype).eval()
    for p in Am.parameters():
        p.requires_grad_(False)
    return Am


def r18_out(A, x, M, rec=None):
    """Masked ResNet-18 (hand-written, non-inplace ReLU). M: {'stream{s}': (N, C), 'inner{s}{b}': (N, C)}."""
    import wide18
    h = A.maxpool(torch.relu(A.bn1(A.conv1(x))))                          # stage-1 stream (stem output)
    for s, layer in enumerate(wide18.layers(A)):
        for b, bl in enumerate(layer):
            src = s if (b > 0 or s == 0) else s - 1                       # which stage's stream this block reads
            hin = h * M[f"stream{src}"][:, :, None, None]
            if rec is not None:
                rec.unit(f"stream{src}", h, hin, 1, (2, 3))
            i = torch.relu(bl.bn1(bl.conv1(hin)))
            im = i * M[f"inner{s}{b}"][:, :, None, None]
            if rec is not None:
                rec.unit(f"inner{s}{b}", i, im, 1, (2, 3))
            skip = h if bl.downsample is None else bl.downsample(hin)
            h = torch.relu(bl.bn2(bl.conv2(im)) + skip)
    hin = h * M["stream3"][:, :, None, None]
    if rec is not None:
        rec.unit("stream3", h, hin, 1, (2, 3))
    return A.fc(torch.flatten(A.avgpool(hin), 1))


# ---------------------------------------------------------------- one scoring pass
def run_pass(net, model, X, R, kept, perm=None, grad=True, bs=None, dev=DEV, dtype=torch.float32):
    """model: Td (CLIP dict on dev) or a prep_r18 copy. kept: {name: bool (units,)}.
    grad=True: returns scores {name: (units,)} (rule, or null when perm is given) and outputs.
    grad=False: returns (None, outputs)."""
    fwd = clip_out if net == "clip" else r18_out
    bs = bs or BS[net]
    S = {k: torch.zeros(len(v), device=dev, dtype=dtype) for k, v in kept.items()} if grad else None
    outs = []
    for a in range(0, len(X), bs):
        xb = X[a:a + bs].to(dev).to(dtype)
        B = len(xb)
        M = {k: v.to(dev).to(dtype)[None].expand(B, -1).clone().requires_grad_(grad) for k, v in kept.items()}
        if not grad:
            with torch.no_grad():
                outs.append(fwd(model, xb, M).detach().float().cpu())
            continue
        rec = Rec({k: p.to(dev) for k, p in perm.items()}) if perm is not None else None
        with torch.enable_grad():
            o = fwd(model, xb, M, rec)
            (R[a:a + B].to(dev).to(dtype) * o).sum().backward()
        for k, m in M.items():
            G = m.grad
            if rec is not None:
                G = (G - rec.G["_D"] + rec.G["_Dp"]) if k == "stream" else rec.G[k]
            S[k] += G.pow(2).sum(0)
        outs.append(o.detach().float().cpu())
    if grad:
        S = {k: (v / len(X)).float().cpu() for k, v in S.items()}
    return S, torch.cat(outs)


def probe_err(net, o, ot):
    d = o - ot
    if net == "r18":
        d = d - d.mean(1, keepdim=True)
    return float(d.pow(2).sum(1).mean())


# ---------------------------------------------------------------- greedy
def _greedy(net, model, X, R, sets, size_fn, sched, perm):
    kept = {k: torch.ones(n, dtype=torch.bool) for k, n in sets.items()}
    step = {k: torch.full((n,), len(sched), dtype=torch.long) for k, n in sets.items()}
    sc = {k: torch.zeros(n) for k, n in sets.items()}
    prt = parts(net, list(sets))
    nr = min(N_READ[net], len(X))
    score1, o_t, readout = None, None, []
    for i, f in enumerate(sched):
        t0 = time.time()
        S, o = run_pass(net, model, X, R, kept, perm, True)
        if i == 0:
            score1 = {k: v.clone() for k, v in S.items()}
            o_t = o
        row = dict(step=i, f=f, kept={k: int(v.sum()) for k, v in kept.items() if k in ("stream", "stream0", "head0", "mlp0", "inner00")},
                   all=probe_err(net, o, o_t), all_n=len(X), all_sub=probe_err(net, o[:nr], o_t[:nr]), sub_n=nr)
        if i > 0:
            for part, names in prt.items():
                kp = {k: (kept[k] if k in names else torch.ones_like(kept[k])) for k in kept}
                _, op = run_pass(net, model, X[:nr], None, kp, None, False)
                row[part] = probe_err(net, op, o_t[:nr])
        for k in kept:
            sc[k][kept[k]] = S[k][kept[k]]
        if i < len(sched) - 1:
            tgt = size_fn(sched[i + 1])
            for k in kept:
                idx = kept[k].nonzero().squeeze(1)
                n_rm = len(idx) - tgt[k]
                if n_rm <= 0:
                    continue
                o_ = torch.sort(S[k][idx], stable=True).indices
                rm = idx[o_[:n_rm]]
                kept[k][rm] = False
                step[k][rm] = i
        row["seconds"] = round(time.time() - t0, 1)
        readout.append(row)
        log(f"{net} step {i} f={f}: probe err all {row['all']:.4g}", {p: round(row[p], 4) for p in prt if p in row}, f"{row['seconds']}s")
    return dict(step=step, score=sc, score1=score1, readout=readout)


def order(net, model, X=None, seed=1, null=False, stop=None, sched=None, tag=None, save=True):
    """The nested removal order, cached per (net, input, seed, null, stop). model: T dict (CLIP) or A (ResNet, CPU)."""
    tag = tag or input_tag(X)
    sched = tuple(sched or [f for f in SCHED if stop is None or f >= stop - 1e-9])
    name = f"p5rank_{net}_{tag}_s{seed}" + ("_null" if null else "") + (f"_to{stop:g}" if stop else "") + f"_v{VERSION}.pt"
    path = os.path.join(CACHE, name)
    if path in _MEM:
        return _MEM[path]
    if save and os.path.exists(path):
        C = torch.load(path, weights_only=True)
        _MEM[path] = C
        return C
    X = (default_input() if X is None else X).float()
    t0 = time.time()
    if net == "clip":
        mdl = vitgen.to(model, DEV)
        R = probe(net, len(X), seed, model["proj"].shape[1])
    else:
        mdl = prep_r18(model)
        R = probe(net, len(X), seed, model.fc.weight.shape[0])
    size_fn = sizes_fn(net, model)
    sets = size_fn(1.0)
    perm = make_perm(sets, seed) if null else None
    log(f"building order {name} on {len(X)} inputs, {len(sched)} steps")
    res = _greedy(net, mdl, X, R, sets, size_fn, sched, perm)
    C = dict(version=VERSION, net=net, input=tag, seed=seed, null=null, stop=stop, sched=list(sched), n=len(X),
             perm=perm, photo_use="none (weights + generic input only)", seconds=round(time.time() - t0, 1), **res)
    del mdl
    if DEV == "mps":
        torch.mps.empty_cache()
    if save:
        os.makedirs(CACHE, exist_ok=True)
        tmp = path + ".tmp"
        torch.save(C, tmp)
        os.replace(tmp, path)
    _MEM[path] = C
    log(f"order done in {C['seconds']}s -> {path if save else '(not saved)'}")
    return C


def score(net, model, X=None, seed=1):
    """Step-1 (unmasked teacher) P5 scores per set, from the cached order (P0's check imports this)."""
    return order(net, model, X, seed)["score1"]


# ---------------------------------------------------------------- order -> kept sets
def rank(C, name):
    """Integer rank per unit: 0 = removed first. Key = (step at removal, score at removal)."""
    st, sc = C["step"][name].numpy(), C["score"][name].numpy()
    o = np.lexsort((sc, st))
    r = np.empty(len(o), dtype=np.int64)
    r[o] = np.arange(len(o))
    return torch.from_numpy(r)


def keep(C, name, k):
    st = C["step"][name]
    n = len(st)
    if k >= n:
        return torch.arange(n)
    n_last = int((st == len(C["sched"])).sum())
    if C["stop"] is not None and k < n_last:
        raise ValueError(f"p5rank: this order stops at f={C['stop']}; size {k} < {n_last} kept at the stop ({name})")
    r = rank(C, name).double()
    return torch.sort(torch.topk(r, k).indices).values


def keep1(C, name, k):
    """One-shot: top k by the step-1 score (row p5rank1)."""
    s = C["score1"][name]
    return torch.arange(len(s)) if k >= len(s) else torch.sort(torch.topk(s, k).indices).values


def kept_at(C, size):
    out = {}
    for k, n in size.items():
        m = torch.zeros(len(C["step"][k]), dtype=torch.bool)
        m[keep(C, k, n)] = True
        out[k] = m
    return out


# ---------------------------------------------------------------- builders
def build_vit(T, f, C, which="all", oneshot=False):
    """which: 'all' (whole cut), or 'stream' / 'heads' / 'mlp' (only that set cut, others full)."""
    pick = keep1 if oneshot else keep
    nb = len(T["blocks"])
    d, h, m = vitgen.sizes(f, T)
    D, H, M = vitgen.sizes(1.0, T)
    d, h, m = (d if which in ("all", "stream") else D), (h if which in ("all", "heads") else H), (m if which in ("all", "mlp") else M)
    return vitgen.cut(T, pick(C, "stream", d), [pick(C, f"head{i}", h) for i in range(nb)], [pick(C, f"mlp{i}", m) for i in range(nb)])


@torch.no_grad()
def build_r18(A, f, C, which="all", oneshot=False):
    """which: 'all' (wide18.cut_from_a), or 'stream' / 'inner' (only that set cut, by zeroing its reads, as
    rules/_split.py does; params reported = the full network's)."""
    import wide18
    pick = keep1 if oneshot else keep
    w = wide18.widths(f)
    if which == "all":
        sc = dict(stream=[], inner=[])
        for s in range(4):
            v = torch.zeros(wide18.BASE[s])
            v[pick(C, f"stream{s}", w[s])] = 1.0
            sc["stream"].append(v)
            row = []
            for b in range(2):
                v = torch.zeros(wide18.BASE[s])
                v[pick(C, f"inner{s}{b}", w[s])] = 1.0
                row.append(v)
            sc["inner"].append(row)
        S, _ = wide18.cut_from_a(A.cpu(), f, sc)
        return S.cpu().eval()
    S = copy.deepcopy(A).cpu().eval()
    L = wide18.layers(S)
    for s in range(4):
        if which == "inner":
            for b in range(2):
                cut = torch.ones(wide18.BASE[s], dtype=torch.bool)
                cut[pick(C, f"inner{s}{b}", w[s])] = False
                L[s][b].conv2.weight[:, cut] = 0
        elif which == "stream":
            cut = torch.ones(wide18.BASE[s], dtype=torch.bool)
            cut[pick(C, f"stream{s}", w[s])] = False
            readers = [L[s][1].conv1]
            if s < 3:
                readers += [L[s + 1][0].conv1, L[s + 1][0].downsample[0]]
            if s == 0:
                readers += [L[0][0].conv1]
            for r in readers:
                r.weight[:, cut] = 0
            if s == 3:
                S.fc.weight[:, cut] = 0
        else:
            raise ValueError(which)
    return S


def make(which="all", null=False, seed=1, oneshot=False):
    """Rule functions with board.py's signatures."""
    def vit(T, f, X=None):
        C = order("clip", T, X, seed=seed, null=null, stop=NULL_STOP["clip"] if null else None)
        return build_vit(T, f, C, which, oneshot)

    def r18(A, f, X=None):
        C = order("r18", A, X, seed=seed, null=null, stop=NULL_STOP["r18"] if null else None)
        return build_r18(A, f, C, which, oneshot)
    return vit, r18


vit, r18 = make()
