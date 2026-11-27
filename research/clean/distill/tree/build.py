# scale: neurons=every node of CLIP ViT-B/16 (36,864 MLP + 144 heads) and ResNet-18 (1,920 inner); net=both; photos=none (generic input only); where=mac
# rung: Q=tree loop 1, whole build per distill/PAGE.md v1; EST=1 min per CLIP build, 5 s ResNet; PERMIT=Shonu approved plan 2026-11-26 late night
"""The tree, built whole (distill/PAGE.md). Line numbers [Ln] refer to the page.

Calculation input: the big network's weights + the generic input X (fanout/cache/*_self_n256_k32_s1.pt). No photos.
  build_clip(T, f, X, loo=None) -> vitgen dict       build_r18(A, f, X, loo=None) -> torchvision-shaped ResNet-18
  loo: None | "fire" | "write" | "rand"   (PAGE leave-one-out rows)
  plant: CHECK ONLY. A list of exact sibling copies was added to the big network; see checks.py.
"""
import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DIST, "fanout"))
import vitgen  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
NS = 8192
HD = 64
torch.set_grad_enabled(False)


# ================================================================== generic: tree, cut, coarse node
def strong(z):
    """[L7] strong-firing level: mean of the top 1% (fallback: std when that is not positive). z: (N, m)."""
    k = max(1, z.shape[0] // 100)
    top = z.topk(k, dim=0).values.mean(0)
    return torch.where(top > 1e-6, top, z.std(0).clamp_min(1e-6))


def chains(parent, ids):
    """[L4] ancestor chain (list of ids, nearest first) of every id in ids. parent: dict id -> id or -1."""
    out = {}
    for i in ids:
        c, p = [], parent[i]
        while p >= 0:
            c.append(p)
            p = parent[p]
        out[i] = c
    return out


def _partition(ids, ch, L):
    """Groups of ids that share an ancestor within L steps (union-find). L = None: all in one group."""
    if L is None:
        return [list(ids)]
    up = {}

    def find(a):
        while up.get(a, a) != a:
            up[a] = up.get(up[a], up[a])
            a = up[a]
        return a

    for i in ids:
        for a in ch[i][:L]:
            ra, ri = find(("a", a)), find(("n", i))
            if ra != ri:
                up[ri] = ra
    g = {}
    for i in ids:
        g.setdefault(find(("n", i)), []).append(i)
    return list(g.values())


def cut(ids, ch, k, strength, read, rng=None):
    """[L5-L6] groups of ids (local index lists) for a block of size k. ids: list of global ids (position = local index).
    strength: (n,) node strength. read: (n, r) read vectors for the closest-read partner. rng: 'rand' row -> random
    groups with the same sizes. Returns (groups as local index lists, info)."""
    n = len(ids)
    loc = {g: j for j, g in enumerate(ids)}
    info = dict(n=n, k=k)
    if k >= n:
        return [[j] for j in range(n)], dict(info, level=0)
    prev = [[g] for g in ids]
    L, maxL = 1, max(len(c) for c in ch.values()) + 1
    while True:
        Lp = L if L <= maxL else None
        P = _partition(ids, ch, Lp)
        if len(P) <= k or Lp is None:
            break
        prev, L = P, L + 1
    info["level"] = L if L <= maxL else "root"
    sup = {}
    for si, grp in enumerate(P):
        for g in grp:
            sup[g] = si
    # merge part of the way: weakest group first, into the closest-read sibling group of the same super group
    groups = [[loc[g] for g in grp] for grp in prev]
    gsup = [sup[prev[i][0]] for i in range(len(prev))]
    rn = F.normalize(read.float(), dim=1)
    st = strength.double()
    alive = list(range(len(groups)))
    gstr = {i: float(st[groups[i]].sum()) for i in alive}
    rep = {i: max(groups[i], key=lambda j: float(st[j])) for i in alive}
    bysup = {}
    for i in alive:
        bysup.setdefault(gsup[i], set()).add(i)
    nmerge = len(groups) - k
    for _ in range(nmerge):
        cand = [i for i in gstr if len(bysup[gsup[i]]) > 1]
        i = min(cand, key=lambda c: gstr[c])
        sib = [j for j in bysup[gsup[i]] if j != i]
        j = max(sib, key=lambda s: float(rn[rep[i]] @ rn[rep[s]]))
        groups[j] += groups[i]
        gstr[j] += gstr[i]
        if st[rep[i]] > st[rep[j]]:
            rep[j] = rep[i]
        bysup[gsup[i]].discard(i)
        del gstr[i]
    out = [groups[i] for i in gstr]
    if rng is not None:                                          # 'rand' row: same sizes, members shuffled
        perm = torch.randperm(n, generator=rng).tolist()
        outr, a = [], 0
        for grp in out:
            outr.append(perm[a:a + len(grp)])
            a += len(grp)
        out = outr
    sizes = sorted((len(g) for g in out), reverse=True)
    info.update(groups=len(out), max_group=sizes[0], singles=sum(s == 1 for s in sizes))
    return out, info


def _thresh(zh, rate):
    """[L8] t with share(zh > t) = rate. zh: (N,). No crossing seen (rate 0 or 1): t = 0, the sum's own zero."""
    N = zh.numel()
    s = zh.sort(descending=True).values
    c = int(round(float(rate) * N))
    if c <= 0 or c >= N:
        return 0.0
    lo, hi = float(s[c]), float(s[c - 1])                     # any t in [lo, hi) gives the rate: take the one nearest 0
    return min(max(0.0, lo), hi - 1e-7 * max(1.0, abs(hi)))


def coarse_unit(Z, M, lvl, act, loo, rep):
    """[L7-L10] one coarse unit from members M (local idx) with teacher pre-activations Z (N, m).
    Returns (read coefficients c (|M|,), bias shift b0, write coefficients w (|M|,)):
      coarse pre-activation = sum_i c_i z_i + b0      coarse write = sum_i w_i write_i."""
    Zm = Z[:, M]
    if loo == "fire":                                          # PAGE row 'fire': strongest member's own read and gate
        c = torch.zeros(len(M))
        c[M.index(rep)] = 1.0
        b0 = 0.0
    else:
        s = lvl[M]
        zh = Zm @ (1.0 / s)
        rate = (Zm > 0).any(1).float().mean()                    # [L8] union rate
        t = _thresh(zh, rate)
        a = float(s.max() / strong(zh[:, None])[0])              # [L9] scale of the read itself (before the gate shift)
        c, b0 = a / s, -a * t
    zc = Zm @ c + b0
    if loo == "write":                                         # PAGE row 'write': strongest member's own write
        w = torch.zeros(len(M))
        w[M.index(rep)] = 1.0
        return c, b0, w
    fire = zc > 0
    kt = max(4, zc.numel() // 100)
    if int(fire.sum()) < kt:                                   # [L10] fires on < 1% of tokens: its top 1%
        fire = zc >= zc.topk(kt).values[-1]
    den = act(zc[fire]).mean()                                 # [L10] GELU means can be negative: keep the sign
    den = den if abs(float(den)) > 1e-8 else torch.tensor(1e-8)
    w = act(Zm[fire]).mean(0) / den
    return c, b0, w


# ================================================================== CLIP adapter
def _centre(W):
    return W - W.mean(0, keepdim=True)


@torch.no_grad()
def clip_stats(T, X, bs=32):
    """[L14] one pass of the big network on X: RMS of every parent signal, NS sampled MLP pre-activations per block."""
    Td = vitgen.to(T, DEV)
    nb = len(T["blocks"])
    n, L = len(X), T["pos"].shape[0]
    g = torch.Generator().manual_seed(3)
    pick = torch.randperm(n * L, generator=g)[:NS]
    emb2, head2, mlp2 = 0.0, [0.0] * nb, [0.0] * nb
    emb1, head1, mlp1 = 0.0, [0.0] * nb, [0.0] * nb
    Z = [[] for _ in range(nb)]
    for a in range(0, n, bs):
        x = X[a:a + bs].to(DEV).float()
        N = len(x)
        loc = (pick[(pick >= a * L) & (pick < (a + N) * L)] - a * L).to(DEV)
        s = vitgen.embed(Td, x)
        emb2 = emb2 + s.pow(2).sum((0, 1))
        emb1 = emb1 + s.sum((0, 1))
        for i, B in enumerate(Td["blocks"]):
            o = heads_out(vitgen.norm(s, B["ln1"]), B)
            head2[i] = head2[i] + o.pow(2).sum((0, 1))
            head1[i] = head1[i] + o.sum((0, 1))
            s = s + F.linear(o, B["o_w"], B["o_b"])
            z = F.linear(vitgen.norm(s, B["ln2"]), B["fc_w"], B["fc_b"])
            Z[i].append(z.flatten(0, 1)[loc].cpu())
            gl = F.gelu(z)
            mlp2[i] = mlp2[i] + gl.pow(2).sum((0, 1))
            mlp1[i] = mlp1[i] + gl.sum((0, 1))
            s = s + F.linear(gl, B["pr_w"], B["pr_b"])
    cnt = n * L
    sd = lambda m2, m1: (m2 / cnt - (m1 / cnt) ** 2).clamp_min(0).sqrt().cpu()  # noqa: E731  [L3] spread, not RMS
    return dict(emb=sd(emb2, emb1), head=[sd(a, b) for a, b in zip(head2, head1)],
                mlp=[sd(a, b) for a, b in zip(mlp2, mlp1)], Z=[torch.cat(z) for z in Z])


def heads_out(y, B):
    N, L, _ = y.shape
    h = B["q_w"].shape[0] // HD
    q = F.linear(y, B["q_w"], B["q_b"]).view(N, L, h, HD).transpose(1, 2)
    k = F.linear(y, B["k_w"], B["k_b"]).view(N, L, h, HD).transpose(1, 2)
    v = F.linear(y, B["v_w"], B["v_b"]).view(N, L, h, HD).transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(N, L, h * HD)


def clip_tree(T, st):
    """[L2-L3] ids, parent, strength, read for CLIP. ids: ('e', d) embed root, ('h', b, h) head, ('m', b, i) MLP."""
    nb, d = len(T["blocks"]), T["cls"].numel()
    H = T["blocks"][0]["q_w"].shape[0] // HD
    gid, names = {}, []

    def new(name):
        gid[name] = len(names)
        names.append(name)
        return gid[name]

    for e in range(d):
        new(("e", e))
    for b in range(nb):
        for h in range(H):
            new(("h", b, h))
        for i in range(T["blocks"][b]["fc_w"].shape[0]):
            new(("m", b, i))
    parent = {gid[("e", e)]: -1 for e in range(d)}
    clarity = {}
    # parent write columns, scaled by RMS, centred over dims (LayerNorm follows)
    Eemb = torch.diag(st["emb"])                                         # (d, d): column e = e_e * rho
    Eemb = _centre(Eemb)
    Ohead = [[_centre(T["blocks"][b]["o_w"][:, h * HD:(h + 1) * HD].float()) * st["head"][b][h * HD:(h + 1) * HD][None]
              for h in range(H)] for b in range(nb)]
    Wmlp = [_centre(T["blocks"][b]["pr_w"].float()) * st["mlp"][b][None] for b in range(nb)]

    def best(scores, cand_ids):
        """running top-2 over score blocks (r, p_j): main parent and clarity (top / second) per reader."""
        v1 = v2 = i1 = None
        off = 0
        for Sj in scores:
            Sj = Sj() if callable(Sj) else Sj
            k = min(2, Sj.shape[1])
            tv, ti = Sj.topk(k, dim=1)
            ti = ti + off
            off += Sj.shape[1]
            if v1 is None:
                v1, i1 = tv[:, 0], ti[:, 0]
                v2 = tv[:, 1] if k > 1 else torch.full_like(v1, -1e30)
                continue
            for c in range(k):
                x, xi = tv[:, c], ti[:, c]
                up = x > v1
                v2 = torch.where(up, v1, torch.maximum(v2, x))
                i1 = torch.where(up, xi, i1)
                v1 = torch.where(up, x, v1)
        return [cand_ids[j] for j in i1.tolist()], v1 / v2.abs().clamp_min(1e-12)

    for b in range(nb):
        B = T["blocks"][b]
        # heads of block b: read = V (64 x d) * ln1.g; candidates embed, heads < b, MLP < b; size only
        Rv = B["v_w"].float() * B["ln1"]["g"].float()[None]
        for h in range(H):
            R = Rv[h * HD:(h + 1) * HD]
            sc, cid = [(R @ Eemb).norm(dim=0)[None]], [gid[("e", e)] for e in range(d)]
            for bb in range(b):
                sc.append(torch.stack([(R @ O).norm() for O in Ohead[bb]])[None])
                cid += [gid[("h", bb, hh)] for hh in range(H)]
                sc.append((R @ Wmlp[bb]).norm(dim=0)[None])
                cid += [gid[("m", bb, i)] for i in range(Wmlp[bb].shape[1])]
            p, cl = best(sc, cid)
            parent[gid[("h", b, h)]] = p[0]
            clarity[gid[("h", b, h)]] = float(cl[0])
        # MLP of block b: read = fc_w * ln2.g; candidates embed, heads <= b (size), MLP < b (signed)
        R = B["fc_w"].float() * B["ln2"]["g"].float()[None]
        sc, cid = [(R @ Eemb).abs()], [gid[("e", e)] for e in range(d)]
        for bb in range(b + 1):
            sc.append(torch.stack([(R @ O).norm(dim=1) for O in Ohead[bb]], 1))
            cid += [gid[("h", bb, hh)] for hh in range(H)]
        for bb in range(b):
            sc.append(lambda bb=bb: R @ Wmlp[bb])
            cid += [gid[("m", bb, i)] for i in range(Wmlp[bb].shape[1])]
        p, cl = best(sc, cid)
        for i in range(R.shape[0]):
            parent[gid[("m", b, i)]] = p[i]
            clarity[gid[("m", b, i)]] = float(cl[i])
    # [L6] strength = RMS x size of the write as the next readers read it
    stren, reads = {}, {}
    for b in range(nb):
        B = T["blocks"][b]
        Rm = B["fc_w"].float() * B["ln2"]["g"].float()[None]                # block b MLP reads (read the heads of b)
        if b + 1 < nb:
            Bn = T["blocks"][b + 1]
            Rn = torch.cat([Bn["v_w"].float() * Bn["ln1"]["g"].float()[None], Bn["fc_w"].float() * Bn["ln2"]["g"].float()[None]])
        else:
            Rn = (T["proj"].float() * T["ln_post"]["g"].float()[:, None]).T     # (512, d) class-token readout
        stren[("h", b)] = torch.stack([(Rm @ O).norm() for O in Ohead[b]])
        stren[("m", b)] = (Rn @ Wmlp[b]).norm(dim=0)
        reads[("h", b)] = (B["v_w"].float() * B["ln1"]["g"].float()[None]).view(H, -1)
        reads[("m", b)] = Rm
    return dict(gid=gid, names=names, parent=parent, clarity=clarity, strength=stren, reads=reads, H=H, nb=nb)


def clip_groups(T, tr, f, loo=None, seed=11, ks=None):
    _, Hk, Mk = vitgen.sizes(f, T)
    ch = chains(tr["parent"], [g for g in range(len(tr["names"])) if tr["names"][g][0] != "e"])
    out, infos = [], []
    for b in range(tr["nb"]):
        res = {}
        for kind, k in (("h", Hk), ("m", Mk)):
            n = tr["strength"][(kind, b)].numel()
            if ks is not None:
                k = ks[(kind, b)]
            ids = [tr["gid"][(kind, b, i)] for i in range(n)]
            rng = torch.Generator().manual_seed(seed + b) if loo == "rand" else None
            grp, info = cut(ids, {g: ch[g] for g in ids}, k, tr["strength"][(kind, b)], tr["reads"][(kind, b)], rng)
            res[kind] = grp
            infos.append(dict(block=b, kind=kind, **info))
        out.append(res)
    return out, infos


@torch.no_grad()
def build_clip(T, f, X, loo=None, st=None, tr=None, verbose=True, ks=None):
    st = st or clip_stats(T, X)
    tr = tr or clip_tree(T, st)
    groups, infos = clip_groups(T, tr, f, loo, ks=ks)
    W = copy.deepcopy(T)
    W["blocks"] = []
    for b, B in enumerate(T["blocks"]):
        strength = tr["strength"]
        # ---- MLP [L7-L10]
        Z = st["Z"][b].float()
        lvl = strong(Z)
        fw, fb, pw = B["fc_w"].float(), B["fc_b"].float(), B["pr_w"].float()
        nfw, nfb, npw = [], [], []
        for M in groups[b]["m"]:
            if len(M) == 1:                                                # [L12]
                nfw.append(fw[M[0]]); nfb.append(fb[M[0]]); npw.append(pw[:, M[0]])
                continue
            rep = max(M, key=lambda j: float(strength[("m", b)][j]))
            c, b0, w = coarse_unit(Z, M, lvl, F.gelu, loo, rep)
            nfw.append(c @ fw[M]); nfb.append(c @ fb[M] + b0); npw.append(pw[:, M] @ w)
        # ---- heads [L11]
        r = lambda t, h: t[h * HD:(h + 1) * HD]  # noqa: E731
        hq, hk, hv, hqb, hkb, hvb, ow = [], [], [], [], [], [], []
        ob = B["o_b"].float().clone()
        for M in groups[b]["h"]:
            rep = max(M, key=lambda j: float(strength[("h", b)][j]))
            if len(M) == 1 or loo == "write":                             # [L12] own head; 'write' row: strongest head only
                h = M[0] if len(M) == 1 else rep
                hq.append(r(B["q_w"], h)); hk.append(r(B["k_w"], h)); hv.append(r(B["v_w"], h))
                hqb.append(r(B["q_b"], h)); hkb.append(r(B["k_b"], h)); hvb.append(r(B["v_b"], h))
                ow.append(B["o_w"][:, h * HD:(h + 1) * HD])
                continue
            OV = sum(B["o_w"][:, h * HD:(h + 1) * HD].double() @ r(B["v_w"], h).double() for h in M)
            U, S, Vt = torch.linalg.svd(OV, full_matrices=False)
            sq = S[:HD].sqrt()
            ow.append((U[:, :HD] * sq[None]).float())
            hv.append((sq[:, None] * Vt[:HD]).float())
            hvb.append(torch.zeros(HD))
            hq.append(r(B["q_w"], rep)); hk.append(r(B["k_w"], rep))
            hqb.append(r(B["q_b"], rep)); hkb.append(r(B["k_b"], rep))
            for h in M:                                                    # V biases -> output bias (softmax rows sum to 1)
                ob = ob + B["o_w"][:, h * HD:(h + 1) * HD].float() @ r(B["v_b"], h).float()
        W["blocks"].append(dict(
            ln1=B["ln1"], ln2=B["ln2"], q_w=torch.cat(hq), k_w=torch.cat(hk), v_w=torch.cat(hv),
            q_b=torch.cat(hqb), k_b=torch.cat(hkb), v_b=torch.cat(hvb), o_w=torch.cat(ow, 1), o_b=ob,
            fc_w=torch.stack(nfw), fc_b=torch.stack(nfb), pr_w=torch.stack(npw, 1), pr_b=B["pr_b"]))
    if verbose:
        mg = [i for i in infos if i["kind"] == "m"]
        hg = [i for i in infos if i["kind"] == "h"]
        print(f"[tree clip f={f:g} loo={loo}] MLP level {[i['level'] for i in mg]} max group {[i.get('max_group', 1) for i in mg]}"
              f" | heads level {[i['level'] for i in hg]}", flush=True)
    W["_infos"] = infos
    return W


# ================================================================== ResNet-18 adapter
def r18_blocks(A):
    return [blk for layer in (A.layer1, A.layer2, A.layer3, A.layer4) for blk in layer]


def _bn_affine(bn):
    sc = bn.weight.float() / (bn.running_var.float() + bn.eps).sqrt()
    return sc, bn.bias.float() - sc * bn.running_mean.float()


@torch.no_grad()
def r18_stats(A, X, bs=32):
    """[L14] one pass of A on X: RMS of stem channels and every inner channel; NS sampled inner pre-activations per block."""
    Ad = copy.deepcopy(A).to(DEV).eval()
    blks = r18_blocks(Ad)
    n = len(X)
    g = torch.Generator().manual_seed(3)
    stem2, inner2, Z, cnt = 0.0, [0.0] * len(blks), [[] for _ in blks], [0] * len(blks)
    stem1, inner1 = 0.0, [0.0] * len(blks)
    cells = None
    picks = [None] * len(blks)
    for a in range(0, n, bs):
        x = X[a:a + bs].to(DEV).float()
        s = Ad.maxpool(Ad.relu(Ad.bn1(Ad.conv1(x))))
        stem2 = stem2 + s.pow(2).sum((0, 2, 3))
        stem1 = stem1 + s.sum((0, 2, 3))
        cs = s.shape[0] * s.shape[2] * s.shape[3]
        for i, blk in enumerate(blks):
            z = blk.bn1(blk.conv1(s))
            Hh, Ww = z.shape[2], z.shape[3]
            if picks[i] is None:
                picks[i] = torch.randperm(n * Hh * Ww, generator=g)[:NS]
            p = picks[i]
            loc = (p[(p >= a * Hh * Ww) & (p < (a + len(x)) * Hh * Ww)] - a * Hh * Ww).to(DEV)
            Z[i].append(z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])[loc].cpu())
            inner2[i] = inner2[i] + F.relu(z).pow(2).sum((0, 2, 3))
            inner1[i] = inner1[i] + F.relu(z).sum((0, 2, 3))
            cnt[i] += z.shape[0] * Hh * Ww
            s = blk(s)
        cells = (cells or 0) + cs
    sd = lambda m2, m1, c: (m2 / c - (m1 / c) ** 2).clamp_min(0).sqrt().cpu()  # noqa: E731  [L3] spread, not RMS
    return dict(stem=sd(stem2, stem1, cells), inner=[sd(v, w, c) for v, w, c in zip(inner2, inner1, cnt)],
                Z=[torch.cat(z) for z in Z])


def r18_tree(A, st):
    A = A.cpu()
    blks = r18_blocks(A)
    gid, names = {}, []
    for c in range(64):
        gid[("e", c)] = len(names); names.append(("e", c))
    for b, blk in enumerate(blks):
        for i in range(blk.conv1.out_channels):
            gid[("m", b, i)] = len(names); names.append(("m", b, i))
    parent = {gid[("e", c)]: -1 for c in range(64)}
    clarity = {}
    # writes in the stream right after their block (Cout,), x RMS
    wr, ds = [], []
    for b, blk in enumerate(blks):
        sc, _ = _bn_affine(blk.bn2)
        wr.append((blk.conv2.weight.float().sum((2, 3)) * sc[:, None]) * st["inner"][b][None])   # (Cout, k)
        if blk.downsample is not None:
            dsc, _ = _bn_affine(blk.downsample[1])
            ds.append(blk.downsample[0].weight.float()[:, :, 0, 0] * dsc[:, None])
        else:
            ds.append(None)

    def to_block(Wc, src, dst):
        """write columns made after block src (src=-1: stem) as read by block dst."""
        for u in range(src + 1, dst):
            if ds[u] is not None:
                Wc = ds[u] @ Wc
        return Wc

    root = torch.diag(st["stem"])
    stren, reads = {}, {}
    for b, blk in enumerate(blks):
        sc1, _ = _bn_affine(blk.bn1)
        R = blk.conv1.weight.float().sum((2, 3)) * sc1[:, None]                                # (k, Cin)
        S = [R @ to_block(root, -1, b)]
        cid = [gid[("e", c)] for c in range(64)]
        for bb in range(b):
            S.append(R @ to_block(wr[bb], bb, b))
            cid += [gid[("m", bb, i)] for i in range(wr[bb].shape[1])]
        S = torch.cat(S, 1)
        top2 = S.topk(2, dim=1)
        for i in range(R.shape[0]):
            parent[gid[("m", b, i)]] = cid[int(top2.indices[i, 0])]
            clarity[gid[("m", b, i)]] = float(top2.values[i, 0] / top2.values[i, 1].abs().clamp_min(1e-12))
        if b + 1 < len(blks):
            nb_ = blks[b + 1]
            scn, _ = _bn_affine(nb_.bn1)
            Rn = nb_.conv1.weight.float().sum((2, 3)) * scn[:, None]
        else:
            Rn = A.fc.weight.float()
        stren[("m", b)] = (Rn @ wr[b]).norm(dim=0)
        reads[("m", b)] = (blk.conv1.weight.float() * sc1[:, None, None, None]).flatten(1)
    return dict(gid=gid, names=names, parent=parent, clarity=clarity, strength=stren, reads=reads, nb=len(blks))


def r18_groups(A, tr, f, loo=None, seed=11, ks=None):
    blks = r18_blocks(A)
    ch = chains(tr["parent"], [g for g in range(len(tr["names"])) if tr["names"][g][0] != "e"])
    out, infos = [], []
    for b, blk in enumerate(blks):
        n = blk.conv1.out_channels
        k = max(1, round(f * n)) if ks is None else ks[("m", b)]
        ids = [tr["gid"][("m", b, i)] for i in range(n)]
        rng = torch.Generator().manual_seed(seed + b) if loo == "rand" else None
        grp, info = cut(ids, {g: ch[g] for g in ids}, k, tr["strength"][("m", b)], tr["reads"][("m", b)], rng)
        out.append(grp)
        infos.append(dict(block=b, kind="m", **info))
    return out, infos


@torch.no_grad()
def build_r18(A, f, X, loo=None, st=None, tr=None, verbose=True, ks=None):
    A = A.cpu().eval()
    st = st or r18_stats(A, X)
    tr = tr or r18_tree(A, st)
    groups, infos = r18_groups(A, tr, f, loo, ks=ks)
    S = copy.deepcopy(A).cpu().eval()
    for b, (blk, sblk) in enumerate(zip(r18_blocks(A), r18_blocks(S))):
        Z = st["Z"][b].float()
        lvl = strong(Z)
        sc1, sh1 = _bn_affine(blk.bn1)
        W1, W2 = blk.conv1.weight.float(), blk.conv2.weight.float()
        k = len(groups[b])
        c1 = nn.Conv2d(W1.shape[1], k, 3, blk.conv1.stride, 1, bias=False)
        b1 = nn.BatchNorm2d(k)
        c2 = nn.Conv2d(k, W2.shape[0], 3, 1, 1, bias=False)
        nw1, nsc, nsh, nw2 = [], [], [], []
        for M in groups[b]:
            if len(M) == 1:                                                # [L12]
                j = M[0]
                nw1.append(W1[j]); nsc.append(sc1[j]); nsh.append(sh1[j]); nw2.append(W2[:, j])
                continue
            rep = max(M, key=lambda j: float(tr["strength"][("m", b)][j]))
            c, b0, w = coarse_unit(Z, M, lvl, F.relu, loo, rep)
            # coarse pre = sum c_i (sc1_i conv_i + sh1_i) + b0  -> one conv row with unit scale
            nw1.append((c[:, None, None, None] * sc1[M][:, None, None, None] * W1[M]).sum(0))
            nsc.append(torch.tensor(1.0)); nsh.append(c @ sh1[M] + b0)
            nw2.append((W2[:, M] * w[None, :, None, None]).sum(1))
        c1.weight.copy_(torch.stack(nw1))
        c2.weight.copy_(torch.stack(nw2, 1))
        b1.weight.copy_(torch.stack(nsc)); b1.bias.copy_(torch.stack(nsh))
        b1.running_mean.zero_(); b1.running_var.fill_(1.0 - b1.eps)
        sblk.conv1, sblk.bn1, sblk.conv2 = c1, b1, c2
    S.eval()
    if verbose:
        print(f"[tree r18 f={f:g} loo={loo}] level {[i['level'] for i in infos]} max group {[i.get('max_group', 1) for i in infos]}",
              flush=True)
    S._infos = infos
    return S
