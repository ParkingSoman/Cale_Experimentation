"""Routes and uses of any InceptionV1 unit (lab helpers, exploration moves O9 and O10).

Build of unit u at (x, p): the votes c_j(x, p) = sum over kernel offsets o of K_u[j, o] h_j(x, p + o), one number per
builder j (a channel of the input of u's final conv: the block input for the 1x1 branch, the bottleneck for the 3x3
and 5x5 branches, the max-pooled input for the pool branch). Route = the lead builder argmax_j c_j.
Use of u at (x, p): the gated reads g_r = W_{r,u} 1[z_r(x, p') > 0] for every reader r of the next block that reads u
through a 1x1 conv (its 1x1 branch and its two bottlenecks), with p' = p, or p // 2 when a stride-2 max pool sits
between the blocks. (The next block's pool branch is left out.) Readers are rescaled to |(w_r, b_r)| = 1 (reduced
form), so the use does not depend on how a reader is scaled.
"""
import numpy as np
import torch
import torch.nn.functional as F

from lib import BLOCKS, DEV, _POOL_BEFORE, _pool, prep, run

WIDTHS = {"3a": (64, 128, 32, 32), "3b": (128, 192, 96, 64), "4a": (192, 208, 48, 64), "4b": (160, 224, 64, 64),
          "4c": (128, 256, 64, 64), "4d": (112, 288, 64, 64), "4e": (256, 320, 128, 128),
          "5a": (256, 320, 128, 128), "5b": (384, 384, 128, 128)}
NAMES = ("1x1", "3x3", "5x5", "pool")


def branch_of(blk, ch):
    lo = 0
    for nm, w in zip(NAMES, WIDTHS[blk]):
        if ch < lo + w:
            return nm, ch - lo
        lo += w
    raise ValueError(ch)


def builders(m, blk, ch, h_in):
    """Per-builder vote maps (B x n_builders x H x W) of unit ch of block blk, given the block input h_in."""
    br, k = branch_of(blk, ch)
    g = lambda s: getattr(m, f"mixed{blk}_{s}")
    if br == "1x1":
        src, conv, pad = h_in, g("1x1_pre_relu_conv"), 0
    elif br == "pool":
        src = g("pool")(F.pad(h_in, (1, 1, 1, 1), value=float('-inf')), kernel_size=(3, 3), stride=(1, 1), padding=0,
                        ceil_mode=False)
        conv, pad = g("pool_reduce_pre_relu_conv"), 0
    else:
        bn = "3x3_bottleneck" if br == "3x3" else "5x5_bottleneck"
        src = g(bn)(g(bn + "_pre_relu_conv")(h_in))
        conv, pad = g(br + "_pre_relu_conv"), (1 if br == "3x3" else 2)
    K = conv.weight[k]  # n_builders x kh x kw
    s = F.pad(src, (pad, pad, pad, pad)) if pad else src
    return F.conv2d(s, K[:, None], groups=K.shape[0])


def reader_convs(m, blk):
    nxt = BLOCKS[BLOCKS.index(blk) + 1]
    return nxt, [getattr(m, f"mixed{nxt}_{s}_pre_relu_conv") for s in ("1x1", "3x3_bottleneck", "5x5_bottleneck")]


def reader_weights(m, blk, ch):
    """Rescaled read weights of unit ch (a global channel of blk) by every 1x1-type reader of the next block."""
    _, convs = reader_convs(m, blk)
    ws = []
    for c in convs:
        W = c.weight[:, :, 0, 0]
        n = torch.sqrt((W ** 2).sum(1) + c.bias ** 2)
        ws.append(W[:, ch] / n)
    return torch.cat(ws)


def target_samples(m, X, img_ix, blk, ch, n_pos, rng, batch=50):
    vals = []
    with torch.no_grad():
        for s in range(0, len(img_ix), batch):
            h = run(m, prep(X[img_ix[s:s + batch]]))[blk][:, ch]
            vals.append(h.reshape(h.shape[0], -1).cpu().numpy())
    v = np.concatenate(vals)
    G = int(round(np.sqrt(v.shape[1])))
    if v.sum() <= 0:  # the unit is not live on these images (Definition D43)
        return None, v
    w = v.ravel() / v.sum()
    pick = rng.choice(len(w), n_pos, replace=True, p=w)
    return np.array([(img_ix[k // v.shape[1]], (k % v.shape[1]) // G, k % G) for k in pick]), v


def build_route_use(m, X, picks, blk, ch, batch=50):
    """For each picked (image, row, col): the vote vector over builders, and the gated read vector over readers."""
    nxt, convs = reader_convs(m, blk)
    rw = reader_weights(m, blk, ch)
    U, Guse = [], []
    with torch.no_grad():
        for s in range(0, len(picks), batch):
            pk = picks[s:s + batch]
            r = run(m, prep(X[pk[:, 0]]))
            votes = builders(m, blk, ch, r["in_" + blk])
            ar = torch.arange(len(pk), device=DEV)
            rows = torch.tensor(pk[:, 1], device=DEV)
            cols = torch.tensor(pk[:, 2], device=DEV)
            U.append(votes[ar, :, rows, cols].cpu().numpy())
            h = r[blk]
            if nxt in _POOL_BEFORE:
                h = _pool(m, _POOL_BEFORE[nxt], h)
                rows2, cols2 = torch.clamp(rows // 2, max=h.shape[2] - 1), torch.clamp(cols // 2, max=h.shape[3] - 1)
            else:
                rows2, cols2 = rows, cols
            z = torch.cat([c(h) for c in convs], 1)[ar, :, rows2, cols2]  # pre-activations of the readers
            Guse.append((rw[None, :] * (z > 0).float()).cpu().numpy())
    return np.concatenate(U), np.concatenate(Guse)


def packing_index(U, Gu, min_group=10, seed=0):
    """pi = 1 - |sum_g w_g m_g|^2 / (sum_g w_g |m_g|)^2 over routes g (lead builder), with route means m_g of the use.
    Squared norms are estimated without bias from two random halves (m_g^(1) . m_g^(2)); routes with fewer than
    min_group samples are pooled into one route. Returns pi, the route sizes, and the route means."""
    lead = np.argmax(U, 1)
    ids, cnt = np.unique(lead, return_counts=True)
    lead = np.where(np.isin(lead, ids[cnt < min_group]), -1, lead)
    rng = np.random.default_rng(seed)
    half = rng.random(len(lead)) < 0.5
    n = len(lead)
    tot1, tot2, s_norm, sizes, means = 0, 0, 0.0, {}, {}
    for g in np.unique(lead):
        ix = lead == g
        a, b = ix & half, ix & ~half
        if a.sum() == 0 or b.sum() == 0:
            continue
        m1, m2 = Gu[a].mean(0), Gu[b].mean(0)
        w = ix.sum() / n
        s_norm += w * np.sqrt(max(m1 @ m2, 0.0))
        tot1, tot2 = tot1 + w * m1, tot2 + w * m2
        sizes[int(g)] = int(ix.sum())
        means[int(g)] = Gu[ix].mean(0)
    num = max(float(np.dot(tot1, tot2)), 0.0)
    return (1 - num / s_norm ** 2 if s_norm > 0 else 0.0), sizes, means


def read_detail(m, X, picks, blk, ch, batch=50):
    """For each picked (image, row, col): the readers' pre-activations z_r at the reading position, the value of u that
    those readers see there (after the pool, if any), and the rescaled read weights (one entry per 1x1-type reader)."""
    nxt, convs = reader_convs(m, blk)
    rw = reader_weights(m, blk, ch)
    norms = torch.cat([torch.sqrt((c.weight[:, :, 0, 0] ** 2).sum(1) + c.bias ** 2) for c in convs])
    Z, Hs = [], []
    with torch.no_grad():
        for s in range(0, len(picks), batch):
            pk = picks[s:s + batch]
            h = run(m, prep(X[pk[:, 0]]))[blk]
            ar = torch.arange(len(pk), device=DEV)
            rows = torch.tensor(pk[:, 1], device=DEV)
            cols = torch.tensor(pk[:, 2], device=DEV)
            if nxt in _POOL_BEFORE:
                h = _pool(m, _POOL_BEFORE[nxt], h)
                rows, cols = torch.clamp(rows // 2, max=h.shape[2] - 1), torch.clamp(cols // 2, max=h.shape[3] - 1)
            z = torch.cat([c(h) for c in convs], 1)[ar, :, rows, cols] / norms[None, :]  # rescaled readers
            Z.append(z.cpu().numpy())
            Hs.append(h[ar, ch, rows, cols].cpu().numpy())
    return np.concatenate(Z), np.concatenate(Hs), rw.cpu().numpy()
