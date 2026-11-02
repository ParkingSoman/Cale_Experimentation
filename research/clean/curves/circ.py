"""Part VI shared tools. factor_build(K): the factorised formula of Prop. 16.2 for any 3920 x 3920 metric K on the family's
signals (2 inputs x 10 members x 196 positions), activity moments from the 2000 train images of x5f (rng 5 set, rng 11
order); returns the build and its per-image dev-5000 KL. Also the 4a pieces of the transport identity (Prop. 18.5)."""
import numpy as np, torch
from inception import *
import b3_train
from family import Data, kl, pool3, IN4A

m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"])
keep = [i for i in range(480) if i not in fam]; R = keep + [480]
TR = np.sort(np.random.default_rng(5).choice(D.tr, 2000, replace=False)); ORDER = np.random.default_rng(11).permutation(TR)
WF = [getattr(m, n).weight.detach()[:, fam, 0, 0] for n in IN4A]           # on DEV
C3 = m.mixed4a_3x3_pre_relu_conv; C5 = m.mixed4a_5x5_pre_relu_conv
N1, N3, N5, NP = WF[0].shape[0], C3.weight.shape[0], C5.weight.shape[0], WF[3].shape[0]   # 4a output blocks


def family_inputs(b):
    with torch.no_grad():
        x = pool3(F.relu(pre_3b(m, inputs_3b(m, to_3a(m, prep(D.X[b])))[0])))
        p = F.max_pool2d(F.pad(x, (1, 1, 1, 1), value=float('-inf')), 3, 1)
    return x, p


def readers_post(x, p):
    return [F.relu(F.conv2d(inp, getattr(m, n).weight, getattr(m, n).bias)) for n, inp in zip(IN4A, [x, x, x, p])]


def gates(post):
    with torch.no_grad():
        return (C3(F.pad(post[1], (1, 1, 1, 1))) > 0).float(), (C5(F.pad(post[2], (2, 2, 2, 2))) > 0).float()


def rest_from_4a(hh):
    for n in ["mixed4b", "mixed4c", "mixed4d", "mixed4e"]: hh = mixed(m, n, hh)
    hh = pool3(hh)
    for n in ["mixed5a", "mixed5b"]: hh = mixed(m, n, hh)
    return m.softmax2_pre_activation_matmul(F.avg_pool2d(hh, 7).reshape(-1, 1024))


def grads_4a(post, K=10):
    """J at the 4a outputs for the top-K classes (renormalised probs pk), as in x5_kbar: (B, K, 508, 14, 14)."""
    with torch.no_grad():
        a, b, cc, d = post
        H = torch.cat([a, F.relu(C3(F.pad(b, (1, 1, 1, 1)))), F.relu(C5(F.pad(cc, (2, 2, 2, 2)))), d], 1)
    H = H.requires_grad_(True); lg = rest_from_4a(H)
    top = lg.detach().softmax(1).topk(K, 1); pk = top.values / top.values.sum(1, keepdim=True)
    J = [torch.autograd.grad(lg.gather(1, top.indices[:, c:c + 1]).sum(), H, retain_graph=c < K - 1)[0] for c in range(K)]
    return torch.stack(J, 1), pk


def transport(JH, G3, G5):
    """Family-signal gradients g = A^T J (Prop. 18.5) with gates G3, G5 on the 3x3 / 5x5 branches: (B, K, 3920)."""
    B, K = JH.shape[:2]; J = JH.flatten(0, 1)
    j0 = J[:, :N1]; j3 = J[:, N1 + N3 + N5:]
    j1 = F.conv_transpose2d(G3.repeat_interleave(K, 0) * J[:, N1:N1 + N3] if G3.shape[0] == B else G3 * J[:, N1:N1 + N3], C3.weight, padding=1)
    j2 = F.conv_transpose2d(G5.repeat_interleave(K, 0) * J[:, N1 + N3:N1 + N3 + N5] if G5.shape[0] == B else G5 * J[:, N1 + N3:N1 + N3 + N5], C5.weight, padding=2)
    gx = sum(torch.einsum("of,nopq->nfpq", WF[i], t) for i, t in enumerate([j0, j1, j2])); gp = torch.einsum("of,nopq->nfpq", WF[3], j3)
    return torch.cat([gx, gp], 1).reshape(B, K, 3920)


def centred(g, pk):
    return ((g - (pk[:, :, None] * g).sum(1, keepdim=True)) * pk[:, :, None].sqrt()).reshape(-1, g.shape[-1])


def diag_metric(w):
    """sum over 4a units u and centres q of w(u, q) row(u, q) row(u, q)^T, rows = A's rows with gates folded into w.
    w: dict of (units, 14, 14) weights for the blocks '1x1', '3x3', '5x5', 'pool' (CPU float64). Returns 3920 x 3920."""
    Wc = lambda t: t.detach().cpu().double()
    k3 = torch.einsum("ubyx,bf->ufyx", Wc(C3.weight), Wc(WF[1])); k5 = torch.einsum("ubyx,bf->ufyx", Wc(C5.weight), Wc(WF[2]))
    Kd = torch.zeros(2, 10, 14, 14, 2, 10, 14, 14, dtype=torch.float64)
    for blk, Wd, s in [("1x1", Wc(WF[0]), 0), ("pool", Wc(WF[3]), 1)]:          # pointwise: same position only
        G = torch.einsum("uyx,uf,ug->fgyx", w[blk], Wd, Wd)
        for y in range(14):
            for x in range(14): Kd[s, :, y, x, s, :, y, x] += G[:, :, y, x]
    for blk, k, h in [("3x3", k3, 1), ("5x5", k5, 2)]:
        S = 2 * h + 1
        for cy in range(14):
            for cx in range(14):
                V = torch.zeros(k.shape[0], 10, 14, 14, dtype=torch.float64)
                for a in range(S):
                    for b_ in range(S):
                        y, x = cy + a - h, cx + b_ - h
                        if 0 <= y < 14 and 0 <= x < 14: V[:, :, y, x] = k[:, :, a, b_]
                V = V.reshape(k.shape[0], 1960)
                Kd[0, :, :, :, 0] += (V.T @ (w[blk][:, cy, cx, None] * V)).reshape(10, 14, 14, 10, 14, 14)
    return Kd.reshape(3920, 3920)


_Z = None
def _activity():
    global _Z
    if _Z is None:
        Zs = []
        for i in range(0, 2000, 8):
            x, p = family_inputs(np.sort(ORDER[i:i + 8]))
            Zs.append(torch.stack([torch.cat([t, torch.ones_like(t[:, :1])], 1).flatten(2).transpose(1, 2) for t in (x, p)], 1).cpu())
        _Z = torch.cat(Zs)                                                      # (2000, 2, 196, 481)
    return _Z


def build_from_lambda(L):
    fm = b3_train.FamilyRA(m, fam, 1); fm.add_rb(m); fm.add_ra(m); WFc = [w.cpu() for w in WF]
    with torch.no_grad():
        for cc in fm.cols: cc.zero_()
        for j in range(4): fm.dW[j].copy_((WFc[j] @ L[:, :470]) / fm.cscale[j]); fm.cbp[j].copy_((WFc[j] @ L[:, 470]) / 0.1)
    return fm.to(DEV)


def factor_lambda(K):
    Kb = K.float().reshape(2, 10, 196, 2, 10, 196).to(DEV); Z = _activity()
    H = torch.zeros(10, 481, 10, 481, dtype=torch.float64); Hg = torch.zeros(10, 481, 10, 481, device=DEV)
    for n in range(2000):
        z = Z[n].to(DEV); T = torch.einsum("sfptgq,tqb->sfpgb", Kb, z); Hg += torch.einsum("spa,sfpgb->fagb", z, T)
        if n % 64 == 63: H += Hg.cpu().double(); Hg.zero_()
    H += Hg.cpu().double(); H /= 2000
    A = H[:, R][:, :, :, R].reshape(4710, 4710); v = sum(H[:, R, g, fam[g]] for g in range(10)).reshape(4710)
    return torch.linalg.solve(A + 1e-6 * torch.trace(A) / 4710 * torch.eye(4710, dtype=torch.float64), v).reshape(10, 471).float()


def per_image(fm, ix=None, bs=100):
    ix = D.dv if ix is None else ix; out = []
    with torch.no_grad():
        for i in range(0, len(ix), bs):
            bb = ix[i:i + bs]; lg = b3_train.forward_b3(m, fm, inputs_3b(m, to_3a(m, prep(D.X[bb]))))
            out.append(kl(torch.from_numpy(D.T[bb].astype(np.float32)).to(DEV), lg).cpu().numpy())
    return np.concatenate(out)


def factor_build(K, tag):
    fm = build_from_lambda(factor_lambda(K)); torch.save(fm.state_dict(), f"cache/{tag}.pt"); r = per_image(fm)
    print(f"{tag}: {r.mean():.4f} (se {r.std() / np.sqrt(len(r)):.4f})", flush=True); return r


def paired(name, r, refs):
    for k, v in refs.items():
        d = r - v; print(f"  {name} minus {k:10s} {d.mean():+.4f} (paired se {d.std() / np.sqrt(len(d)):.4f})", flush=True)


REFS = lambda: {"level0p": np.load("cache/x2_gated.npz")["b_level0_p"], "wmetric": np.load("cache/x5h_wbuild.npz")["wmetric_n2000"],
                "kbar": np.load("cache/x5f_factor.npz")["factor_n2000"]}
