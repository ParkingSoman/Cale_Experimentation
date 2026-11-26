"""P1-P3 round 2 (2026-11-26): OR-merge on CLIP MLP-only. Stream and heads stay full width (teacher weights).

Object (picked by): the MLP neurons of every block. Kept set = the floor's MLP order (rules/noisecut scores on X).
A deleted neuron joins one kept neuron of the same block (its group). The write of every row is P4's fold
(rules/p4write.fold_map, slice anchor, closed loop, MLP sites only), solved on the student's OWN merged firings.
So the write rule is held fixed across rows; only the tree (partner), the read and the gate change.

TREE (one forest per block, the same for all sizes)
  parents p = every upstream write column into the stream: embed dims (e_d), o_w columns of blocks 0..b, pr_w
  columns of blocks 0..b-1. Columns centred over dims (LN follows).
  vote v_i[p] = (fc_w[i] * ln2.g) . w~_p * rho_p        tree="r1rms": "R1 weighted by noise RMS" (rho = RMS of the
                                                          parent's scalar on X); tree="r1w": pure weights (rho = 1)
  parent pi(i) = argmax_p v_i[p]. Order = floor score. pointer h(j) = argmax over STRONGER k with pi(k) = pi(j) of
  cos(v_j, v_k); none: argmax over stronger k of v_k[pi(j)]. tree="rand": h(j) = uniform random stronger unit (seed 11).
  At size f a deleted unit climbs h until it reaches a kept unit.
READ (merge=True): zhat_M = sum_{i in M} (z_i - mu_i)/sd_i, divided by its own sd S_M (z on the teacher stream, X).
GATE  gate="or":  rate = P_X(max_{i in M} z_i > 0)     gate="and": rate = P_X(z_k > 0) (the kept unit's own rate)
  t_M = (1 - rate) quantile of zhat_M (sort, CPU). z_M = sd_k (zhat_M - t_M).
  merge=False, gate="or": GATE-ONLY row: kept unit's own read, threshold moved to its group's union rate.
  gate=None: no change to the reads (row a, the floor + P4 fold).
PHOTO USE: none. Weights + generic input X (statistics on 8,192 random tokens per block; fold on all X tokens).
"""
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
FAN = os.path.dirname(HERE)
sys.path.insert(0, FAN)
sys.path.insert(0, os.path.dirname(FAN))
import vitgen  # noqa: E402
from rules import noisecut, p4write  # noqa: E402

DEV = noisecut.DEV
BS = 32
NS = 8192
_C = {}


@torch.no_grad()
def _teacher_stats(T, X):
    """One teacher pass on X: per block a token sample of ln2's output y (NS x 768), and parent RMS lists."""
    key = id(X)
    if key in _C:
        return _C[key]
    Td = vitgen.to(T, DEV)
    nb = len(T["blocks"])
    g = torch.Generator().manual_seed(3)
    n, L = len(X), 197
    pick = torch.randperm(n * L, generator=g)[:NS]
    ys = [[] for _ in range(nb)]
    emb2, head2, mlp2 = 0.0, [0.0] * nb, [0.0] * nb
    cnt = 0
    for a in range(0, n, BS):
        x = X[a:a + BS].to(DEV).float()
        s = vitgen.embed(Td, x)
        N = len(x)
        loc = pick[(pick >= a * L) & (pick < (a + N) * L)] - a * L
        emb2 = emb2 + s.pow(2).sum((0, 1))
        for i, B in enumerate(Td["blocks"]):
            o = p4write._heads_fire(s, B)
            head2[i] = head2[i] + o.pow(2).sum((0, 1))
            s = s + F.linear(o, B["o_w"], B["o_b"])
            y = vitgen.norm(s, B["ln2"])
            ys[i].append(y.flatten(0, 1)[loc.to(DEV)].cpu())
            gl = F.gelu(F.linear(y, B["fc_w"], B["fc_b"]))
            mlp2[i] = mlp2[i] + gl.pow(2).sum((0, 1))
            s = s + F.linear(gl, B["pr_w"], B["pr_b"])
        cnt += N * L
    out = dict(y=[torch.cat(v) for v in ys], emb=(emb2 / cnt).sqrt().cpu(),
               head=[(h / cnt).sqrt().cpu() for h in head2], mlp=[(m / cnt).sqrt().cpu() for m in mlp2])
    _C[key] = out
    return out


@torch.no_grad()
def _forest(T, st, bi, score, tree):
    """pointer h (3072,) for block bi. score = floor MLP score (higher = stronger)."""
    B = T["blocks"][bi]
    m = B["fc_w"].shape[0]
    rank = torch.empty(m, dtype=torch.long)
    rank[torch.argsort(score, descending=True)] = torch.arange(m)
    stronger = rank[None, :] < rank[:, None]                       # [j, k]: k stronger than j
    if tree == "rand":
        g = torch.Generator().manual_seed(11 + bi)
        u = torch.rand(m, m, generator=g)
        h = torch.where(stronger, u, torch.full_like(u, -1.0)).argmax(1)
        top = rank == 0
        h[top] = torch.arange(m)[top]
        return h, None
    d = T["cls"].numel()
    cols, rhos = [torch.eye(d)], [st["emb"]]
    for b in range(bi + 1):
        cols.append(T["blocks"][b]["o_w"].float())
        rhos.append(st["head"][b])
    for b in range(bi):
        cols.append(T["blocks"][b]["pr_w"].float())
        rhos.append(st["mlp"][b])
    Wc = torch.cat(cols, 1)
    Wc = Wc - Wc.mean(0, keepdim=True)
    rho = torch.cat(rhos) if tree == "r1rms" else torch.ones(Wc.shape[1])
    Wc = (Wc * rho[None]).to(DEV)
    R = (B["fc_w"].float() * B["ln2"]["g"].float()[None]).to(DEV)
    V = R @ Wc                                                     # (m, P)
    pi = V.argmax(1)
    Vn = V / V.norm(dim=1, keepdim=True).clamp_min(1e-12)
    C = (Vn @ Vn.T).cpu()
    Vp = V[:, pi].cpu()                                            # Vp[k, j] = v_k[pi(j)]
    del V, Vn, Wc
    pi = pi.cpu()
    same = (pi[None, :] == pi[:, None]) & stronger
    s1 = torch.where(same, C, torch.full_like(C, -9.0))
    h1 = s1.argmax(1)
    has = same.any(1)
    s2 = torch.where(stronger, Vp.T, torch.full_like(C, -1e30))
    h2 = s2.argmax(1)
    h = torch.where(has, h1, h2)
    top = rank == 0
    h[top] = torch.arange(m)[top]
    return h, dict(n_parents=int(pi.unique().numel()), frac_sibling=float(has.float().mean()))


def _owner(h, kept_mask):
    g = h.clone()
    for _ in range(len(h)):
        nxt = torch.where(kept_mask[g], g, h[g])
        if torch.equal(nxt, g):
            break
        g = nxt
    own = torch.where(kept_mask, torch.arange(len(h)), g)
    assert bool(kept_mask[own].all())
    return own


def _qcol(Z, rate, fallback):
    """Threshold that leaves rate * N samples above it, per column of Z (N x k), by sort on the CPU: the midpoint
    between the last sample below and the first above. Rate 0 or 1 (no crossing seen on X): the fallback value
    (the kept unit's own natural threshold in the same units), so a group of one is the identity."""
    N = Z.shape[0]
    Zs = Z.sort(0).values
    c = (rate * N).round().long()
    lo = (N - c - 1).clamp(0, N - 1)
    hi = (N - c).clamp(0, N - 1)
    t = 0.5 * (Zs.gather(0, lo[None]) + Zs.gather(0, hi[None])).squeeze(0)
    edge = (c <= 0) | (c >= N)
    return torch.where(edge, fallback, t), int(edge.sum())


@torch.no_grad()
def merged_reads(T, f, X, tree="r1rms", gate="or", merge=True, info=None):
    """Returns per block (fc_w (m_k x d), fc_b (m_k,)) for the kept MLP neurons, and the kept index lists."""
    st = _teacher_stats(T, X)
    sc = p4write._vit_scores(T, X)
    _, _, mk_n = vitgen.sizes(f, T)
    mk = [noisecut.top(s, mk_n) for s in sc["mlp"]]
    out = []
    for bi, B in enumerate(T["blocks"]):
        fw, fb = B["fc_w"].float(), B["fc_b"].float()
        kept = mk[bi]
        if gate is None:
            out.append((fw[kept], fb[kept]))
            continue
        m = fw.shape[0]
        kmask = torch.zeros(m, dtype=torch.bool)
        kmask[kept] = True
        A = torch.zeros(m, len(kept))
        finfo = None
        if tree == "single":                                       # row b: groups of one, deleted units dropped
            A[kept, torch.arange(len(kept))] = 1.0
        else:
            key = ("forest", tree, bi, id(X))
            if key not in _C:
                _C[key] = _forest(T, st, bi, sc["mlp"][bi], tree)
            h, finfo = _C[key]
            own = _owner(h, kmask)
            col = torch.full((m,), -1, dtype=torch.long)
            col[kept] = torch.arange(len(kept))
            A[torch.arange(m), col[own]] = 1.0                     # member i -> column of its owner
        y = st["y"][bi].float()
        Z = y @ fw.T + fb                                          # (NS, m)
        mu, sd = Z.mean(0), Z.std(0).clamp_min(1e-6)
        if gate == "or":
            rate = (((Z > 0).float() @ A) > 0).float().mean(0)
        elif gate == "and":
            rate = (Z[:, kept] > 0).float().mean(0)
        else:
            raise ValueError(gate)
        if merge:
            Zh = ((Z - mu) / sd) @ A
            S = Zh.std(0).clamp_min(1e-6)
            Zh = Zh / S
            sk = sd[kept]
            t, nedge = _qcol(Zh, rate, -mu[kept] / (sk * S))
            w = (sk / S)[:, None] * (A.T @ (fw / sd[:, None]))
            b = sk * ((A.T @ ((fb - mu) / sd)) / S - t)
        else:                                                      # gate-only: own read, threshold at the rate
            Zk = Z[:, kept]
            t, nedge = _qcol(Zk, rate, torch.zeros(len(kept)))
            w, b = fw[kept], fb[kept] - t
        out.append((w, b))
        if info is not None:
            gs = A.sum(0)
            info.append(dict(block=bi, k=len(kept), max_group=int(gs.max()), own_rate=float((Z[:, kept] > 0).float().mean()),
                             rate=float(rate.mean()), nedge=nedge, **(finfo or {})))
    return out, mk


@torch.no_grad()
def build(T, f, X=None, tree="r1rms", gate="or", merge=True, null=False, tag=""):
    X = p4write._x_vit(X)
    info = []
    reads, mk = merged_reads(T, f, X, tree, gate, merge, info)
    D, H, _ = vitgen.sizes(1.0, T)
    sd = torch.arange(D)
    hk = [torch.arange(H) for _ in T["blocks"]]
    W = vitgen.cut(T, sd, hk, mk)
    for B, (w, b) in zip(W["blocks"], reads):
        B["fc_w"], B["fc_b"] = w.to(B["fc_w"].dtype), b.to(B["fc_b"].dtype)
    # P4 fold on the merged firings (closed loop, slice anchor), MLP sites only. Heads = teacher.
    Td, Wd = vitgen.to(T, DEV), vitgen.to(W, DEV)
    nX = len(X)
    sls = p4write._slices(nX)
    t = torch.cat([vitgen.embed(Td, X[sl].to(DEV).float()) for sl in sls])
    s = t.clone()
    for bi in range(len(T["blocks"])):
        BT, Bs = Td["blocks"][bi], Wd["blocks"][bi]
        for sl in sls:
            t[sl] = t[sl] + F.linear(p4write._heads_fire(t[sl], BT), BT["o_w"], BT["o_b"])
            s[sl] = s[sl] + F.linear(p4write._heads_fire(s[sl], BT), BT["o_w"], BT["o_b"])
        n, kept = BT["pr_w"].shape[1], mk[bi]
        cutc = torch.ones(n, dtype=torch.bool)
        cutc[kept] = False
        cutc = cutc.nonzero().flatten().to(DEV)
        mom = p4write._Mom(n, len(kept))
        for sl in sls:
            aT = p4write._mlp_fire(t[sl], BT).flatten(0, 1)
            if null:
                aT = aT.clone()
                aT[:, cutc] = p4write._mlp_fire(t[p4write._roll(sl, nX).to(DEV)], BT).flatten(0, 1)[:, cutc]
            mom.add(aT, p4write._mlp_fire(s[sl], Bs).flatten(0, 1))
        G, _ = p4write.fold_map(mom.Sx / mom.N, mom.Ss / mom.N, kept, mom.N, "slice")
        Bs["pr_w"] = (BT["pr_w"].cpu().double() @ G).float().to(DEV)
        for sl in sls:
            t[sl] = t[sl] + F.linear(p4write._mlp_fire(t[sl], BT), BT["pr_w"], BT["pr_b"])
            s[sl] = s[sl] + F.linear(p4write._mlp_fire(s[sl], Bs), Bs["pr_w"], Bs["pr_b"])
    out = vitgen.to(Wd, "cpu")
    del Td, Wd, t, s
    if DEV == "mps":
        torch.mps.empty_cache()
    if info:
        mg = sum(r["max_group"] for r in info) / len(info)
        print(f"[r2_p1p3 {tag} f={f:g}] mean max group {mg:.1f}; own rate {sum(r['own_rate'] for r in info)/len(info):.3f}"
              f" -> target rate {sum(r['rate'] for r in info)/len(info):.3f}; edge units {sum(r['nedge'] for r in info)}"
              + (f"; sibling frac {sum(r.get('frac_sibling', 0) for r in info)/len(info):.3f}" if 'frac_sibling' in info[0] else ""),
              flush=True)
    return out
