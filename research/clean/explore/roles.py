"""Build and use vectors of a unit on its target (lab helpers for exploration moves O9 and later).

For a unit u (a channel of a block's 1x1 branch, or of a block whose pre-ReLU is a 1x1 conv of the block input) at
input x and position p:
  build  U(x, p) = the votes W[u, j] h_j(x, p) of the units j of the layer below (the 1x1 conv's input), a vector;
  use    D(x, p) = M(x) dy/dh_u(x, p), with M = P_s diag(s), s = sqrt(softmax(y)), P_s = I - s s^T, so that
         |M delta|^2 = delta^T (diag q - q q^T) delta (the second order of the KL loss, Fact F5). Sketched with K
         random directions r_k: D_k = r_k^T M dy/dh_u (a Johnson-Lindenstrauss sketch of D).
The target of u: positions (x, p) drawn with probability proportional to h_u(x, p).
"""
import numpy as np
import torch
import torch.nn.functional as F

from lib import DEV, mixed, prep, run, to_3a, _pool, _POOL_BEFORE, BLOCKS

BRANCH_1x1 = {"3a": 64, "3b": 128, "4a": 192, "4b": 160, "4c": 128, "4d": 112, "4e": 256, "5a": 256, "5b": 384}


def forward_from(m, blk, h):
    """Continue the network from a block output to the logits."""
    i = BLOCKS.index(blk)
    for name in BLOCKS[i + 1:]:
        if name in _POOL_BEFORE:
            h = _pool(m, _POOL_BEFORE[name], h)
        h = mixed(m, "mixed" + name, h)
    hp = F.avg_pool2d(h, kernel_size=(7, 7), stride=(1, 1))
    return m.softmax2_pre_activation_matmul(hp.reshape(-1, 1024))


def sample_target(m, X, img_ix, blk, ch, n_pos, rng, batch=50):
    """Draw n_pos (image, position) pairs from the unit's target over the given images."""
    vals = []
    with torch.no_grad():
        for s in range(0, len(img_ix), batch):
            h = run(m, prep(X[img_ix[s:s + batch]]))[blk][:, ch]
            vals.append(h.reshape(h.shape[0], -1).cpu().numpy())
    v = np.concatenate(vals)  # images x positions
    w = v.ravel() / v.sum()
    pick = rng.choice(len(w), n_pos, replace=True, p=w)
    G = int(round(np.sqrt(v.shape[1])))
    return [(int(img_ix[k // v.shape[1]]), int((k % v.shape[1]) // G), int(k % G)) for k in pick], G


def next_block_sum(m, blk, h):
    """Output of the block after blk, summed over positions (one number per unit of the next block)."""
    i = BLOCKS.index(blk)
    name = BLOCKS[i + 1]
    if name in _POOL_BEFORE:
        h = _pool(m, _POOL_BEFORE[name], h)
    return mixed(m, "mixed" + name, h).sum((2, 3))


def build_and_use(m, X, picks, blk, ch, K=64, seed=0, batch=25, use="next"):
    """U (n x n_below) and the sketched D (n x K) at the picked (image, row, col).
    use="next": D = d(next block's outputs, summed over positions)/dh_u: which readers take the vote here.
    use="output": D = M dy/dh_u (the effect on the output distribution, Fact F5)."""
    assert ch < BRANCH_1x1[blk], "build vectors are written for 1x1-branch units"
    conv = getattr(m, f"mixed{blk}_1x1_pre_relu_conv")
    Wu = conv.weight[ch, :, 0, 0]
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(K, 1008, generator=g).to(DEV) / np.sqrt(K)
    Us, Ds = [], []
    for s in range(0, len(picks), batch):
        pk = picks[s:s + batch]
        x = prep(X[[p[0] for p in pk]])
        with torch.no_grad():
            r = run(m, x)
            hin = r["in_" + blk]
            hb = r[blk]
        rows = torch.tensor([p[1] for p in pk], device=DEV)
        cols = torch.tensor([p[2] for p in pk], device=DEV)
        ar = torch.arange(len(pk), device=DEV)
        Us.append((Wu[None, :] * hin[ar, :, rows, cols]).cpu().numpy())
        hb = hb.clone().requires_grad_(True)
        D = []
        if use == "next":
            nx = next_block_sum(m, blk, hb)
            Rn = torch.randn(K, nx.shape[1], generator=g).to(DEV) / np.sqrt(K)
            for k in range(K):
                gk = torch.autograd.grad((Rn[k][None, :] * nx).sum(), hb, retain_graph=(k < K - 1))[0]
                D.append(gk[ar, ch, rows, cols])
            Ds.append(torch.stack(D, 1).detach().cpu().numpy())
            continue
        y = forward_from(m, blk, hb)
        q = torch.softmax(y, 1).detach()
        sq = q.sqrt()
        for k in range(K):
            # r_k^T M dy with M = P_s diag(s): the row vector r_k^T P_s diag(s) = s * (r_k - (r_k . s) s)
            rk = R[k][None, :]
            v = sq * (rk - (rk * sq).sum(1, keepdim=True) * sq)
            gk = torch.autograd.grad((v * y).sum(), hb, retain_graph=(k < K - 1))[0]
            D.append(gk[ar, ch, rows, cols])
        Ds.append(torch.stack(D, 1).detach().cpu().numpy())
    return np.concatenate(Us), np.concatenate(Ds)


def unit_rows(A, eps=1e-12):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.maximum(n, eps)


def dcor_u(A, B):
    """Bias-corrected distance correlation (Szekely and Rizzo 2014, U-centring) of the rows of A and B (Euclidean)."""
    def ucenter(Z):
        n = Z.shape[0]
        d = np.sqrt(np.maximum(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1), 0))
        r = d.sum(1) / (n - 2)
        t = d.sum() / ((n - 1) * (n - 2))
        Uc = d - r[:, None] - r[None, :] + t
        np.fill_diagonal(Uc, 0)
        return Uc
    a, b = ucenter(A), ucenter(B)
    n = A.shape[0]
    f = lambda x, y: (x * y).sum() / (n * (n - 3))
    vab, vaa, vbb = f(a, b), f(a, a), f(b, b)
    return float(vab / np.sqrt(vaa * vbb)) if vaa > 0 and vbb > 0 else 0.0


def route_index(U, D, min_group=5):
    """Packing index of a unit on its target: routes g = the lead builder argmax_j U_j (the unit below with the largest
    vote). pi = 1 - |E D|^2 / sum_g P(g) |E[D | g]|^2, estimated without bias: |E[D|g]|^2 by the mean of D_i . D_j
    over pairs i != j inside g, |E D|^2 by the mean over all pairs i != j. Routes with fewer than min_group samples are
    pooled into one 'other' route. Returns (pi, route sizes, lead builders)."""
    g = np.argmax(U, 1)
    ids, cnt = np.unique(g, return_counts=True)
    small = ids[cnt < min_group]
    g = np.where(np.isin(g, small), -1, g)
    n = len(D)
    G = D @ D.T
    np.fill_diagonal(G, 0)
    tot = G.sum() / (n * (n - 1))
    between = 0.0
    sizes = {}
    for k in np.unique(g):
        ix = np.where(g == k)[0]
        if len(ix) < 2:
            continue
        sub = G[np.ix_(ix, ix)]
        between += len(ix) / n * sub.sum() / (len(ix) * (len(ix) - 1))
        sizes[int(k)] = len(ix)
    return (1 - tot / between if between > 0 else 0.0), sizes
