"""B5 reason step 8: exact averaged (factorised) reader-level objective vs the per-image objective, by samples.
L_fact(B) = 1/2 E_x E_x' sum_c (G_c(x') . e_B(x))^2 (x' metric images, x activity images, independent);
L_img(B) = 1/2 E_x sum_c (G_c(x) . e_B(x))^2 (same image). e_B = change in reader pre-activations (368 x 14 x 14)."""
import numpy as np, torch, copy, time
from circ import *
t0 = time.time()
exec(open("c7_reader.py").read().split("# pass 1: window")[0].split("DD = 6")[1].split("\n", 1)[1])   # sizes, NR, reader_grads
Xm = np.sort(np.random.default_rng(6).choice(np.setdiff1d(D.tr, TR), 500, replace=False)); Xa = np.sort(ORDER[:500])
G = torch.cat([reader_grads(Xm[i:i + 20]).flatten(1).half() for i in range(0, 500, 20)])      # (5000, 72128)
print(f"G {tuple(G.shape)} ({time.time() - t0:.0f}s)", flush=True)
def load_state(path):
    fm = b3_train.FamilyRA(m, fam, 1); fm.add_rb(m); fm.add_ra(m); fm.load_state_dict(torch.load(path, map_location="cpu")); return fm.to(DEV)
S = load_state("cache/x5f_factor_n2000.pt"); T = b3_train.load_b3(m, "B3-k0-factorlocal")
Srem = copy.deepcopy(S); cs = [float(c) for c in S.cscale]; off = np.cumsum([0] + sizes)
dW = torch.cat([(T.dW[j] - S.dW[j]).detach() * cs[j] for j in range(4)]).cpu().double(); Wc = torch.cat([w.cpu() for w in WF]).double()
rem = (dW - Wc @ torch.linalg.lstsq(Wc, dW).solution).float().to(DEV)
with torch.no_grad():
    for j in range(4): Srem.dW[j] += rem[off[j]:off[j + 1]] / cs[j]
builds = {"S": S, "S+rem": Srem, "T": T, "c7_span": load_state("cache/c7_span.pt"), "c7_reader": load_state("cache/c7_reader.pt")}
convs = [getattr(m, n) for n in IN4A]
def pre(x, p, fm=None):
    out = []
    for j, (c, inp) in enumerate(zip(convs, [x, x, x, p])):
        if fm is None: out.append(F.conv2d(inp, c.weight, c.bias)); continue
        W = (c.weight[:, fm.keep, 0, 0] + fm.dW[j] * fm.cscale[j])[:, :, None, None]
        out.append(F.conv2d(inp[:, fm.keep], W, c.bias + fm.colbias[j]))
    return torch.cat(out, 1)
Lf = {k: 0.0 for k in builds}; Li = {k: 0.0 for k in builds}
for i in range(0, 500, 20):
    b = Xa[i:i + 20]; x, p = family_inputs(b); Gi = reader_grads(b).flatten(1)               # (200, 72128) same images
    with torch.no_grad():
        a0 = pre(x, p)
        for k, fm in builds.items():
            e = (a0 - pre(x, p, fm)).flatten(1)                                                # (20, 72128)
            Lf[k] += float(((G.float() @ e.T) ** 2).sum()) / 5000 * 10 / 2                    # mean over x' of sum_c
            Li[k] += float((((Gi.reshape(20, 10, -1) * e[:, None]).sum(2)) ** 2).sum()) / 2
for k in builds: print(f"{k:10s} L_fact {Lf[k] / 500:.4f}   L_img {Li[k] / 500:.4f}", flush=True)
print(f"done ({time.time() - t0:.0f}s)")
