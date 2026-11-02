"""Statement B5 (Part VI): the averaged second-order objective with a READER-level metric, solved over per-reader
re-routings B (368 x 471) by conjugate gradients, and its in-span restriction B = W_F Lambda. FITTED microscope.
Metric: K[(j,p),(j',q)] ~ sqrt(w(p)) K_jj'(q - p) sqrt(w(q)), |q - p| <= 6 cells; K_jj'(d) offset-averaged after whitening
by the per-position window w (from the 2000 Kbar images, rng 5). Activity: windowed offset moments Y^{ss'}(d) of
y = (3b activity, 1) for s in {x, p} (2000 images, rng 11 order as x5f)."""
import numpy as np, torch, time, sys
from circ import *

DD = 6; NO = 2 * DD + 1; t0 = time.time(); torch.set_num_threads(8)
sizes = [w.shape[0] for w in WF]; NR = sum(sizes); typ = np.repeat([0, 0, 0, 1], sizes)        # input type per reader
convs = [getattr(m, n) for n in IN4A]

def reader_grads(b):
    """centred, sqrt(pk)-weighted gradients of the top-10 logits at the reader OUTPUTS: (B * 10, 368, 14, 14)."""
    x, p = family_inputs(b); post = [t.requires_grad_(True) for t in readers_post(x, p)]
    a_, b_, c_, d_ = post
    lg = rest_from_4a(torch.cat([a_, F.relu(C3(F.pad(b_, (1, 1, 1, 1)))), F.relu(C5(F.pad(c_, (2, 2, 2, 2)))), d_], 1))
    top = lg.detach().softmax(1).topk(10, 1); pk = top.values / top.values.sum(1, keepdim=True)
    G = torch.stack([torch.cat(torch.autograd.grad(lg.gather(1, top.indices[:, c:c + 1]).sum(), post, retain_graph=c < 9), 1) for c in range(10)], 1)
    G = (G - (pk[:, :, None, None, None] * G).sum(1, keepdim=True)) * pk[:, :, None, None, None].sqrt()
    return G.flatten(0, 1).detach()

# pass 1: window w(p) = E sum_c sum_j G^2
w = torch.zeros(14, 14, device=DEV)
for i in range(0, 2000, 20): w += (reader_grads(TR[i:i + 20]) ** 2).sum((0, 1))
w = (w / w.mean()).clamp(min=1e-3); sw = w.sqrt()
print(f"window: max/min {float(w.max() / w.min()):.1f}  ({time.time() - t0:.0f}s)", flush=True)

def xcorr_acc(A, Acc):
    """Acc[k] += sum_n conj(FFT A_n)^T FFT A_n per frequency k; A: (n, C, 14, 14) on DEV; Acc: (2, nfreq, C, C)."""
    Fh = torch.fft.rfft2(A, s=(27, 27)).flatten(2).permute(2, 0, 1)                          # (nfreq, n, C)
    Re, Im = Fh.real.contiguous(), Fh.imag.contiguous()
    Acc[0] += Re.transpose(1, 2) @ Re + Im.transpose(1, 2) @ Im
    Acc[1] += Re.transpose(1, 2) @ Im - Im.transpose(1, 2) @ Re

def to_offsets(Acc, n):
    """cross-correlation sum_p a_i(p) a_j(p + d) for |d| <= DD: (NO, NO, C, C), float64 CPU."""
    Fc = torch.complex(Acc[0], Acc[1]).cpu().reshape(27, 14, Acc.shape[2], Acc.shape[3]).permute(2, 3, 0, 1) / n
    R = torch.fft.irfft2(Fc, s=(27, 27))                                                      # (C, C, 27, 27), index d mod 27
    idx = torch.tensor([d % 27 for d in range(-DD, DD + 1)])
    return R[:, :, idx][:, :, :, idx].permute(2, 3, 0, 1).double().contiguous()

# pass 2: reader metric K(d) (whitened by the window)
Acc = torch.zeros(2, 27 * 14, NR, NR, device=DEV)
for i in range(0, 2000, 20): xcorr_acc(reader_grads(TR[i:i + 20]) / sw, Acc)
Kd = to_offsets(Acc, 2000); del Acc
nd = torch.tensor([[(14 - abs(a)) * (14 - abs(b)) for b in range(-DD, DD + 1)] for a in range(-DD, DD + 1)], dtype=torch.float64)
Kd /= nd[:, :, None, None]                                                                    # stationary metric = mean over valid p
print(f"K(d) done; |K(0)| share of sum |K(d)|: {float(Kd[DD, DD].abs().sum() / Kd.abs().sum()):.3f}  ({time.time() - t0:.0f}s)", flush=True)
# pass 3: windowed activity offset moments for y = (x or p activity, 1): channels 0..480 (x), 481..961 (p)
Acc = torch.zeros(2, 27 * 14, 962, 962, device=DEV)
for i in range(0, 2000, 8):
    x, p = family_inputs(np.sort(ORDER[i:i + 8]))
    y = torch.cat([x, torch.ones_like(x[:, :1]), p, torch.ones_like(p[:, :1])], 1) * sw
    xcorr_acc(y, Acc)
Yd = to_offsets(Acc, 2000); del Acc
print(f"Y(d) done  ({time.time() - t0:.0f}s)", flush=True)

Rx = torch.tensor(keep + [480]); Fx = torch.tensor(fam)
Ycols = [(Rx, Fx), (Rx + 481, Fx + 481)]                                                     # (rest, family) columns per type
W = torch.cat([w_.cpu() for w_ in WF]).double()                                               # (368, 10)
T_ = [torch.from_numpy(typ == s) for s in range(2)]
def Yblk(s, t, a, b):                                                                         # Y^{st}(d)[a-cols of s, b-cols of t]
    return Yd[:, :, Ycols[s][a]][:, :, :, Ycols[t][b]]
YRR = {(s, t): Yblk(s, t, 0, 0) for s in range(2) for t in range(2)}
YRF = {(s, t): Yblk(s, t, 0, 1) for s in range(2) for t in range(2)}
KB = {(s, t): Kd[:, :, T_[s]][:, :, :, T_[t]] for s in range(2) for t in range(2)}

def op(B, M=YRR):
    """A(B)_s = sum_t sum_d K^{st}(d) B_t M^{st}(d)^T  (B: 368 x 471 or 368 x 10 for the right-hand side)."""
    out = torch.zeros(NR, M[(0, 0)].shape[2], dtype=torch.float64)
    for (s, t), K in KB.items():
        Bt = B[T_[t]]; acc = torch.zeros(int(T_[s].sum()), out.shape[1], dtype=torch.float64)
        for a in range(NO):
            for b_ in range(NO): acc += (K[a, b_] @ Bt) @ M[(s, t)][a, b_].T
        out[T_[s]] += acc
    return out
rhs = op(W, YRF)                                                                              # sum_d K W Y_RF^T
lam_r = 1e-6 * float(sum(KB[(s, s)][DD, DD].diagonal().sum() * YRR[(s, s)][DD, DD].diagonal().sum() for s in range(2))) / (NR * 471)
A = lambda B: op(B) + lam_r * B
Kdiag = Kd[DD, DD].diagonal().clamp(min=1e-12); Yinv = [torch.linalg.inv(YRR[(s, s)][DD, DD] + 1e-6 * torch.eye(471, dtype=torch.float64) * YRR[(s, s)][DD, DD].trace() / 471) for s in range(2)]
def prec(G):
    out = torch.empty_like(G)
    for s in range(2): out[T_[s]] = (G[T_[s]] / Kdiag[T_[s], None]) @ Yinv[s]
    return out

def cg(Aop, b, P, iters=800, tol=1e-6, tag=""):
    x = torch.zeros_like(b); r = b.clone(); z = P(r); pv = z.clone(); rz = (r * z).sum(); b0 = b.norm()
    for k in range(iters):
        Ap = Aop(pv); al = rz / (pv * Ap).sum(); x += al * pv; r -= al * Ap
        if r.norm() / b0 < tol: break
        z = P(r); rz2 = (r * z).sum(); pv = z + (rz2 / rz) * pv; rz = rz2
        if k % 100 == 0: print(f"  cg {tag} it {k}: rel res {float(r.norm() / b0):.2e}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"  cg {tag} done it {k}: rel res {float(r.norm() / b0):.2e}", flush=True); return x

# in-span: B = W Lambda (one shared Lambda, 10 x 471): operator W^T A(W Lambda)
WtW = W.T @ W
Lam = cg(lambda L: W.T @ A(W @ L), W.T @ rhs, lambda G: torch.linalg.solve(WtW, G), tag="span")
Bfull = cg(A, rhs, prec, tag="reader")
def build(B):
    fm = b3_train.FamilyRA(m, fam, 1); fm.add_rb(m); fm.add_ra(m); off = np.cumsum([0] + sizes)
    with torch.no_grad():
        for cc in fm.cols: cc.zero_()
        for j in range(4):
            Bj = B[off[j]:off[j + 1]].float(); fm.dW[j].copy_(Bj[:, :470] / fm.cscale[j]); fm.cbp[j].copy_(Bj[:, 470] / 0.1)
    return fm.to(DEV)
refs = REFS(); g5 = np.load("cache/c5_gapb.npz"); refs["T"] = g5["T"]; res = {}
for k, B in [("c7_span", W @ Lam), ("c7_reader", Bfull)]:
    fm = build(B); torch.save(fm.state_dict(), f"cache/{k}.pt"); res[k] = per_image(fm)
    print(f"{k}: {res[k].mean():.4f} (se {res[k].std() / np.sqrt(len(res[k])):.4f})", flush=True); paired(k, res[k], refs)
np.savez("cache/c7_reader.npz", **res); torch.save(dict(Kd=Kd.float(), w=w.cpu()), "cache/c7_metric.pt")
print(f"done ({time.time() - t0:.0f}s)")
