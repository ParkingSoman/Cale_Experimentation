"""Phase 5: k-member recipe for ANY 3b family (any branch). Keep members K, drop the rest of family F; 4a's input-conv
weights on the kept channels get the Bayes (or variance-keeping) estimate of the dropped channels' contribution; 4a's
biases get the matching constant. No neuron is retrained.
usage: p5_drop.py <family csv> <keep csv | none> [dev|sealed]"""
import sys, json, numpy as np, torch
from inception import *
from family import Data, IN4A, pool3, kl

def fam_stats(m, F_, D, n=1000, seed=11):
    idx = np.sort(np.random.default_rng(seed).choice(D.tr, n, replace=False)); k = len(F_)
    acc = [[np.zeros(k), np.zeros((k, k)), 0] for _ in range(2)]
    with torch.no_grad():
        for i in range(0, n, 100):
            z = F.relu(pre_3b(m, to_3a(m, prep(D.X[idx[i:i + 100]])))[:, F_]); p1 = pool3(z)
            p2 = F.max_pool2d(F.pad(p1, (1, 1, 1, 1), value=-1e9), 3, 1)
            for a, p in zip(acc, [p1, p2]):
                x = p.permute(0, 2, 3, 1).reshape(-1, k).cpu().numpy().astype(np.float64); a[0] += x.sum(0); a[1] += x.T @ x; a[2] += len(x)
    return [(a[0] / a[2], a[1] / a[2] - np.outer(a[0] / a[2], a[0] / a[2])) for a in acc]

def patched(m, F_, keep, st, mode):
    """Return list of (weight, bias) per 4a input conv after the drop."""
    Ki = [F_.index(c) for c in keep]; Ri = [i for i in range(len(F_)) if i not in Ki]; out = []
    for j, name in enumerate(IN4A):
        conv = getattr(m, name); W = conv.weight.detach().clone(); b = conv.bias.detach().clone()
        mu, C = st[1] if j == 3 else st[0]
        A = W[:, F_, 0, 0].cpu().double().numpy(); AR = A[:, Ri]; add = np.zeros((A.shape[0], len(Ki))); cb = AR @ mu[Ri]
        if mode in ("bayes", "vmk") and Ki:
            B = np.linalg.solve(C[np.ix_(Ki, Ki)], C[np.ix_(Ki, Ri)]); est = AR @ B.T
            if mode == "vmk":
                vR = np.einsum("oi,ij,oj->o", AR, C[np.ix_(Ri, Ri)], AR); vE = np.einsum("oi,ij,oj->o", est, C[np.ix_(Ki, Ki)], est)
                est = est * np.sqrt(vR / np.maximum(vE, 1e-9))[:, None]
            add = est; cb = cb - est @ mu[Ki]
        if mode == "zero": cb = cb * 0
        Wn = W.clone(); Wn[:, [F_[i] for i in Ri]] = 0
        if Ki: Wn[:, keep, 0, 0] += torch.tensor(add, dtype=W.dtype, device=W.device)
        out.append((Wn, b + torch.tensor(cb, dtype=b.dtype, device=b.device)))
    return out

def evaluate_patched(m, pw, D, idx, bs=100):
    orig = [(getattr(m, n).weight.data.clone(), getattr(m, n).bias.data.clone()) for n in IN4A]
    for n, (w, b) in zip(IN4A, pw): getattr(m, n).weight.data.copy_(w); getattr(m, n).bias.data.copy_(b)
    ks, tp = [], []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            bb = idx[i:i + bs]; lg = from_3b(m, F.relu(pre_3b(m, to_3a(m, prep(D.X[bb])))))
            t = torch.from_numpy(D.T[bb].astype(np.float32)).to(DEV); ks.append(kl(t, lg).cpu().numpy()); tp.append(lg[:, D.lab].argmax(1).cpu().numpy() == D.y[bb])
    for n, (w, b) in zip(IN4A, orig): getattr(m, n).weight.data.copy_(w); getattr(m, n).bias.data.copy_(b)
    return np.concatenate(ks), float(np.concatenate(tp).mean())

if __name__ == "__main__":
    F_ = [int(t) for t in sys.argv[1].split(",")]; keep = [] if sys.argv[2] == "none" else [int(t) for t in sys.argv[2].split(",")]
    split_ = sys.argv[3] if len(sys.argv) > 3 else "dev"
    m = load_model(); D = Data(seed=0, ndev=1000 if split_ == "dev" else 0); st = fam_stats(m, F_, D); idx = D.dv if split_ == "dev" else D.se
    for mode in (["zero", "const"] if not keep else ["const", "bayes", "vmk"]):
        k, top = evaluate_patched(m, patched(m, F_, keep, st, mode), D, idx)
        print(f"family {len(F_)} keep {keep} {mode:5s}: {split_} KL {k.mean():.4f} (se {k.std()/np.sqrt(len(k)):.4f}) top-1 {top:.4f}", flush=True)
