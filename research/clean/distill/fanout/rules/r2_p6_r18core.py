"""P6 round 2, ResNet-18 stream: READ MAPS folded into readers (P6 round-1 proposal, L_l). Builder: P6, 2026-11-26.
Object (picked by): every reader of a cut stream channel (block conv1, downsample, fc), P6 proposal.
The cut = the floor's stream-only cut on X (as _p6diag_r18 'plain'). A rotated subspace is not possible on ResNet:
ReLU acts per channel after the add, so the stream has a fixed basis at every block output. The legal "1x1 mixing" is
a per-reader read map: the cut channels are predicted from the kept ones at the reader's own stream point,
x_cut ~ B x_kept + const (ridge least squares on X, centred), and W_cut B is folded into the kept columns
(a 1x1 channel mix commutes with the spatial conv). Parameter count = the plain cut's (no new tensor).
The constant + border remainder is folded exactly in mean (as round-1 'mean'): BN running_mean -= mean of
conv_r(x_cut - B x_kept) on X (an edit of the inference map, never read as data); fc: bias +=.
Modes: map (B from the moments), perm (NULL, same form: B's rows deranged among the cut channels, mean fold kept).
Input X only. Photos: scoring only. BatchNorm statistics are never read.
"""
import copy

import torch

from rules import _p6diag_r18 as d6

DEV = d6.DEV
BS = d6.BS
_M = {}


@torch.no_grad()
def _moments(A, X):
    key = id(X) if X is not None else "pink"
    if _M.get("key") == key:
        return _M["m"]
    Ad = copy.deepcopy(A).to(DEV).eval()
    x = d6._input(X)
    S1, S2, N = {}, {}, {}
    for a in range(0, len(x), BS):
        for k, h in d6._points(Ad, x[a:a + BS].to(DEV).float()).items():
            H = h.permute(0, 2, 3, 1).reshape(-1, h.shape[1]).cpu().double()
            S1[k] = S1.get(k, 0) + H.sum(0)
            S2[k] = S2.get(k, 0) + H.T @ H
            N[k] = N.get(k, 0) + H.shape[0]
    m = {}
    for k in S1:
        mu = S1[k] / N[k]
        m[k] = (mu, S2[k] / N[k] - torch.outer(mu, mu))
    _M["key"], _M["m"] = key, m
    del Ad
    return m


@torch.no_grad()
def build(A, f, X, mode="map", ridge=1e-3):
    import wide18
    sc = d6._scores(A, X)
    mom = _moments(A, X)
    w = wide18.widths(f)
    S = copy.deepcopy(A).cpu().eval()
    cut = []
    for s in range(4):
        keep = wide18.keep_idx(sc["stream"][s], w[s])
        m = torch.ones(wide18.BASE[s], dtype=torch.bool)
        m[keep] = False
        cut.append(m)
    readers = [(s, name, pt) for s in range(4) for name, pt in d6._readers(s)]
    Bs, mus = {}, {}
    for s, name, pt in readers:
        if not cut[s].any():
            continue
        mu, C = mom[pt]
        c, k = cut[s], ~cut[s]
        Ckk = C[k][:, k]
        lam = ridge * float(Ckk.trace()) / int(k.sum())
        B = torch.linalg.solve(Ckk + lam * torch.eye(int(k.sum()), dtype=C.dtype), C[k][:, c]).T   # |cut| x |keep|
        if mode == "perm":
            B = B[d6._derangement(B.shape[0], seed=s)]
        Bs[name] = B.float()
        r2 = 1 - float(torch.trace(C[c][:, c] - B.double() @ C[k][:, c]) / torch.trace(C[c][:, c])) if mode == "map" else float("nan")
        print(f"[r2p6 r18 {mode}] f={f:g} reader {name}: cut-channel variance explained by kept (R2 on X) {r2:.3f}", flush=True)
    # mean remainder per reader, on X, teacher stream: mean of reader(x_cut - B x_kept) (cut slots only)
    Ad = copy.deepcopy(A).to(DEV).eval()
    x = d6._input(X)
    contrib, n = {}, 0
    for a in range(0, len(x), BS):
        pts = d6._points(Ad, x[a:a + BS].to(DEV).float())
        for s, name, pt in readers:
            if name not in Bs:
                continue
            h = pts[pt]
            c, k = cut[s].to(DEV), ~cut[s].to(DEV)
            r = torch.zeros_like(h)
            pred = torch.einsum("ck,nkhw->nchw", Bs[name].to(DEV), h[:, k])
            r[:, c] = h[:, c] - pred
            mod, _ = d6._mods(Ad, name)
            contrib[name] = contrib.get(name, 0) + d6._mean_out(d6._apply(mod, name, r))
        n += len(pts["stem"])
    del Ad
    for s, name, pt in readers:
        mod, bn = d6._mods(S, name)
        if name in Bs:
            c, k = cut[s], ~cut[s]
            Wc = mod.weight[:, c]
            if name == "fc":
                mod.weight[:, k] += Wc @ Bs[name]
                mod.bias += (contrib[name] / n).cpu()
            else:
                mod.weight[:, k] += torch.einsum("ocij,ck->okij", Wc, Bs[name])
                bn.running_mean -= (contrib[name] / n).cpu()
        mod.weight[:, cut[s]] = 0
    if DEV == "mps":
        torch.mps.empty_cache()
    return S.eval()
