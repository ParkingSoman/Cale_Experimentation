"""Bench B3 ruler (NEURON-MATH Part III): coarse units, their write, reader biases AND reader weights on the other 470 3b
channels all trained. Warm start from a B2 checkpoint (or 'k0' for no coarse unit)."""
import sys, numpy as np, torch, torch.nn as nn
import family
from inception import *
from family import Data, train, evaluate, IN4A, pool3
from b2_train import FamilyRB

class FamilyRA(FamilyRB):
    def add_ra(self, m):
        self.dW = nn.ParameterList([nn.Parameter(torch.zeros(getattr(m, n).weight.shape[0], len(self.keep))) for n in IN4A])

def forward_b3(m, fm, ins):
    h = F.relu(pre_3b(m, ins[0]))[:, fm.keep]; c = F.relu(fm.pre(ins))
    x = torch.cat([pool3(h), pool3(c)], 1); p = F.max_pool2d(F.pad(x, (1, 1, 1, 1), value=float('-inf')), 3, 1)
    outs = []
    for j, (name, col, s, inp) in enumerate(zip(IN4A, fm.cols, fm.cscale, [x, x, x, p])):
        conv = getattr(m, name)
        W = torch.cat([conv.weight[:, fm.keep, 0, 0] + fm.dW[j] * s, col * s], 1)[:, :, None, None]
        outs.append(F.conv2d(inp, W, conv.bias + fm.colbias[j]))
    a = m.mixed4a_1x1(outs[0]); b = m.mixed4a_3x3_bottleneck(outs[1]); cc = m.mixed4a_5x5_bottleneck(outs[2])
    d = m.mixed4a_pool_reduce(outs[3])
    b = m.mixed4a_3x3(m.mixed4a_3x3_pre_relu_conv(F.pad(b, (1, 1, 1, 1))))
    cc = m.mixed4a_5x5(m.mixed4a_5x5_pre_relu_conv(F.pad(cc, (2, 2, 2, 2))))
    hh = torch.cat([a, b, cc, d], 1)
    for n in ["mixed4b", "mixed4c", "mixed4d", "mixed4e"]: hh = mixed(m, n, hh)
    hh = pool3(hh)
    for n in ["mixed5a", "mixed5b"]: hh = mixed(m, n, hh)
    return m.softmax2_pre_activation_matmul(F.avg_pool2d(hh, 7).reshape(-1, 1024))

family.forward_family = forward_b3          # train() and evaluate() look this name up at call time

def load_b3(m, tag):
    ck = torch.load(f"cache/ckpt/{tag}.pt", map_location="cpu", weights_only=False)
    fm = FamilyRA(m, ck["fam"], ck["k"], ks=ck.get("ks", 5)); fm.add_rb(m); fm.add_ra(m); fm.load_state_dict(ck["state"])
    return fm.to(DEV)

def formula_k0(m, fam, S):
    """Prop. 11.1 closed form at k = 0: Delta-W = W_F Lambda_rest, Delta-b = W_F Lambda_const (S: 481 x 481 moments)."""
    keep = [i for i in range(480) if i not in fam]; R = keep + [480]; S = torch.as_tensor(S, dtype=torch.float64)
    Srr = S[R][:, R]; Sfr = S[fam][:, R]
    Lam = torch.linalg.solve(Srr + 1e-6 * torch.trace(Srr) / len(R) * torch.eye(len(R), dtype=torch.float64), Sfr.T).T.float()
    fm = FamilyRA(m, fam, 1); fm.add_rb(m); fm.add_ra(m)
    with torch.no_grad():
        for c in fm.cols: c.zero_()
        for j, n in enumerate(IN4A):
            w = getattr(m, n).weight.detach()[:, fam, 0, 0].cpu()
            fm.dW[j].copy_((w @ Lam[:, :470]) / fm.cscale[j]); fm.cbp[j].copy_((w @ Lam[:, 470]) / 0.1)
    return fm

if __name__ == "__main__":
    src, tag = sys.argv[1], sys.argv[2]; steps = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 3e-3; ev = int(sys.argv[5]) if len(sys.argv) > 5 else 125
    pat = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    m = load_model(); D = Data(seed=0, ndev=1000)
    if src in ("k0", "formula") or src.startswith("state:"):   # no coarse unit: one unit with zero write, frozen
        if src.startswith("state:"):                         # start at a saved k = 0 build (e.g. x3c_out_n35000)
            fm = FamilyRA(m, list(np.load("cache/s1_fine.npz")["fam"]), 1); fm.add_rb(m); fm.add_ra(m); fm.load_state_dict(torch.load(f"cache/{src[6:]}.pt", map_location="cpu"))
        elif src == "k0":
            fm = FamilyRA(m, CURVES, 1); fm.add_rb(m); fm.add_ra(m)
            with torch.no_grad():
                for c in fm.cols: c.zero_()
        else:
            fm = formula_k0(m, list(np.load("cache/s1_fine.npz")["fam"]), np.load("cache/r3d_S3b.npy"))
        for p in [fm.read, fm.bias, *fm.cols]: p.requires_grad_(False)
        fm = fm.to(DEV); k, fam = 1, fm.fam
    else:
        ck = torch.load(f"cache/ckpt/{src}.pt", map_location="cpu", weights_only=False)
        fm = FamilyRA(m, ck["fam"], ck["k"], ks=ck.get("ks", 5)); fm.add_rb(m); fm.load_state_dict(ck["state"]); fm.add_ra(m)
        fm = fm.to(DEV); k, fam = ck["k"], ck["fam"]
    print("start dev-1000", evaluate(m, fm, D, D.dv), flush=True)
    opt_params = [p for p in fm.parameters() if p.requires_grad]
    fm.parameters = lambda recurse=True: iter(opt_params)       # train() builds Adam from fm.parameters()
    best, step, hist = train(m, fm, D, lr=lr, steps=steps, eval_every=ev, patience=pat)
    torch.save(dict(state=fm.state_dict(), fam=fam, k=k, ks=5, src=src, ra=True), f"cache/ckpt/{tag}.pt")
    D5 = Data(seed=0); print("dev-5000", evaluate(m, fm, D5, D5.dv), flush=True)
    print("BEST", tag, best, "at step", step, flush=True)
