"""Shared I5 scoring: uncertainty, co-firing vs truth, and the P2 proxy (delete conv2 channels 24-31, refit the write)."""
import torch, torch.nn.functional as F
from shapes import DEV
KEEP, DELE = torch.arange(24, device=DEV), torch.arange(24, 32, device=DEV)

def unc(net, X): p = F.softmax(net(X), 1); return 1 - (p ** 2).sum(1)

def cofiring(net, X, Xtruth):
    f, ft = [net.feats(Z).permute(1, 0, 2, 3).reshape(32, -1) for Z in (X, Xtruth)]; iu = torch.triu_indices(32, 32, 1)
    R, Rt = torch.nan_to_num(torch.corrcoef(f)), torch.corrcoef(ft)
    return float(torch.corrcoef(torch.stack([R[iu[0], iu[1]], Rt[iu[0], iu[1]]]))[0, 1])

def fit(net, X, fisher, w=None):
    w = torch.ones(len(X), device=DEV) if w is None else w / w.mean()
    f = net.feats(X).reshape(len(X), 32, 64); fk = torch.cat([f[:, KEEP], torch.ones_like(f[:, :1])], 1); tgt = f[:, DELE]
    if not fisher:
        A = torch.einsum("n,nkp,njp->kj", w, fk, fk); b = torch.einsum("n,nkp,ndp->kd", w, fk, tgt)
        return torch.linalg.lstsq(A.cpu(), b.cpu(), rcond=1e-7).solution.T.to(DEV)
    Wd = net.fc.weight.reshape(4, 32, 64)[:, DELE]; p = F.softmax(net(X), 1); Fm = torch.diag_embed(p) - p[:, :, None] * p[:, None, :]
    Phi = torch.einsum("odp,nkp->nodk", Wd, fk).reshape(len(X), 4, 8 * 25); y0 = torch.einsum("odp,ndp->no", Wd, tgt)
    A = torch.einsum("n,noi,nop,npj->ij", w, Phi, Fm, Phi); b = torch.einsum("n,noi,nop,np->i", w, Phi, Fm, y0)
    return torch.linalg.lstsq(A.cpu(), b.cpu()[:, None], rcond=1e-7).solution[:, 0].to(DEV).reshape(8, 25)

def score(net, B, Xeval):
    W, bfc = net.fc.weight.reshape(4, 32, 64), net.fc.bias; f = net.feats(Xeval).reshape(len(Xeval), 32, 64)
    fk = torch.cat([f[:, KEEP], torch.ones_like(f[:, :1])], 1); g = f.clone(); g[:, DELE] = torch.einsum("dk,nkp->ndp", B, fk)
    lt = F.log_softmax(torch.einsum("ocp,ncp->no", W, f) + bfc, 1); ls = F.log_softmax(torch.einsum("ocp,ncp->no", W, g) + bfc, 1)
    return (lt.exp() * (lt - ls)).sum(1)
