"""B5 reason check: the averaged reader-level objective for S, S+rem, T, c7_span, c7_reader under the c7 metric K_a and an\nindependent estimate K_b (2000 other train images, rng 6). Same windowed offset activity moments Y(d) as c7."""
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


def metric(imgs):
    w = torch.zeros(14, 14, device=DEV)
    for i in range(0, len(imgs), 20): w += (reader_grads(imgs[i:i + 20]) ** 2).sum((0, 1))
    w = (w / w.mean()).clamp(min=1e-3); sw = w.sqrt(); Acc = torch.zeros(2, 27 * 14, NR, NR, device=DEV)
    for i in range(0, len(imgs), 20): xcorr_acc(reader_grads(imgs[i:i + 20]) / sw, Acc)
    Kd = to_offsets(Acc, len(imgs))
    nd = torch.tensor([[(14 - abs(a)) * (14 - abs(b)) for b in range(-DD, DD + 1)] for a in range(-DD, DD + 1)], dtype=torch.float64)
    return Kd / nd[:, :, None, None], sw
C = torch.load("cache/c7_metric.pt"); Ka = C["Kd"].double(); swa = C["w"].to(DEV).sqrt()
other = np.sort(np.random.default_rng(6).choice(np.setdiff1d(D.tr, TR), 2000, replace=False))
Kb, swb = metric(other); print(f"K_b done; cos(K_a, K_b) {float((Ka * Kb).sum() / Ka.norm() / Kb.norm()):.3f}  ({time.time() - t0:.0f}s)", flush=True)
def Ymom(sw):
    Acc = torch.zeros(2, 27 * 14, 962, 962, device=DEV)
    for i in range(0, 2000, 8):
        x, p = family_inputs(np.sort(ORDER[i:i + 8]))
        xcorr_acc(torch.cat([x, torch.ones_like(x[:, :1]), p, torch.ones_like(p[:, :1])], 1) * sw, Acc)
    return to_offsets(Acc, 2000)
def Vmat(fm):
    cs = [float(c) for c in fm.cscale]; V = torch.zeros(NR, 962, dtype=torch.float64); off = np.cumsum([0] + sizes)
    for j in range(4):
        Bj = torch.cat([fm.dW[j].detach().cpu().double() * cs[j], fm.cbp[j].detach().cpu().double()[:, None] * 0.1], 1)
        o = 0 if j < 3 else 481; rows = slice(off[j], off[j + 1])
        V[rows, o + torch.tensor(fam)] = WF[j].cpu().double(); V[rows, o + torch.tensor(keep + [480])] = -Bj
    return V
def L(V, Kd, Yd):
    tot = 0.0
    for a in range(NO):
        for b_ in range(NO): tot += float((Kd[a, b_] * (V @ Yd[a, b_] @ V.T)).sum())
    return 0.5 * tot
def load_state(path):
    fm = b3_train.FamilyRA(m, fam, 1); fm.add_rb(m); fm.add_ra(m); fm.load_state_dict(torch.load(path, map_location="cpu")); return fm
import copy
S = load_state("cache/x5f_factor_n2000.pt"); T = b3_train.load_b3(m, "B3-k0-factorlocal").cpu()
Srem = copy.deepcopy(S); cs = [float(c) for c in S.cscale]
dW = torch.cat([(T.dW[j] - S.dW[j]).detach() * cs[j] for j in range(4)]).double(); Wc = torch.cat([w.cpu() for w in WF]).double()
rem = dW - Wc @ torch.linalg.lstsq(Wc, dW).solution; off = np.cumsum([0] + sizes)
with torch.no_grad():
    for j in range(4): Srem.dW[j] += (rem[off[j]:off[j + 1]] / cs[j]).float()
builds = {"S": S, "S+rem": Srem, "T": T, "c7_span": load_state("cache/c7_span.pt"), "c7_reader": load_state("cache/c7_reader.pt")}
for tag, Kd, sw in [("K_a", Ka, swa), ("K_b", Kb, swb)]:
    Yd = Ymom(sw); vals = {k: L(Vmat(fm), Kd, Yd) for k, fm in builds.items()}
    print(tag, "  ".join(f"{k} {v:.4f}" for k, v in vals.items()), f"({time.time() - t0:.0f}s)", flush=True)
