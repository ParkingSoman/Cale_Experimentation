"""Write law from the 1x1 failure analysis (pre-registered before testing):
column = Bayes (the honest estimate, adds no unpredictable variance); each 4a input-conv reader's bias is shifted so it
fires (pre-activation > 0) exactly as often as in the intact network, on analysis (train) images.
usage: d8_ratewrite.py <read tag>"""
import sys, numpy as np, torch
from inception import *
from family import Family, Data, evaluate, IN4A, pool3, forward_family
from d4_write import stats, writes

def reader_pre(m, fm, ins, intact):
    """Pre-activations of the four 4a input convs (B, out, 14, 14), intact or crushed."""
    if intact:
        x = pool3(F.relu(pre_3b(m, ins[0])))
        p = F.max_pool2d(F.pad(x, (1, 1, 1, 1), value=float('-inf')), 3, 1)
        return [getattr(m, n)(inp) for n, inp in zip(IN4A, [x, x, x, p])]
    h = F.relu(pre_3b(m, ins[0]))[:, fm.keep]; c = F.relu(fm.pre(ins))
    x = torch.cat([pool3(h), pool3(c)], 1); p = F.max_pool2d(F.pad(x, (1, 1, 1, 1), value=float('-inf')), 3, 1)
    outs = []
    for j, (name, col, s, inp) in enumerate(zip(IN4A, fm.cols, fm.cscale, [x, x, x, p])):
        conv = getattr(m, name); W = torch.cat([conv.weight[:, fm.keep, 0, 0], col * s], 1)[:, :, None, None]
        outs.append(F.conv2d(inp, W, conv.bias + (fm.colbias[j] if fm.colbias is not None else 0)))
    return outs

def rate_shift(m, fm, D, n=200, seed=31):
    idx = np.sort(np.random.default_rng(seed).choice(D.tr, n, replace=False)); I, Nw = [[] for _ in IN4A], [[] for _ in IN4A]
    with torch.no_grad():
        for i in range(0, n, 50):
            ins = inputs_3b(m, to_3a(m, prep(D.X[idx[i:i + 50]])))
            for lst, intact in [(I, True), (Nw, False)]:
                for j, z in enumerate(reader_pre(m, fm, ins, intact)):
                    lst[j].append(z.permute(1, 0, 2, 3).reshape(z.shape[1], -1).cpu())
    shifts = []
    for j in range(4):
        a = torch.cat(I[j], 1); b = torch.cat(Nw[j], 1)
        rate = (a > 0).float().mean(1).clamp(1e-4, 1 - 1e-4)
        q = torch.quantile(b, 1 - rate[:, None].expand(-1, 1).squeeze(1).clamp(0, 1), dim=1) if False else torch.stack([torch.quantile(b[k], 1 - rate[k]) for k in range(b.shape[0])])
        shifts.append((-q).to(DEV))
    return shifts

if __name__ == "__main__":
    m = load_model(); D = Data(seed=0, ndev=1000); fam = list(np.load("cache/s1_fine.npz")["fam"]); tag = sys.argv[1]
    ck = torch.load(f"cache/ckpt/{tag}.pt", map_location="cpu", weights_only=False); fm = Family(m, fam, 1, ks=ck.get("ks", 5)).to(DEV); fm.load_state_dict(ck["state"])
    res, g = writes(m, fam, stats(m, fm, fam, D))
    for wname in ["bayes", "vmr"]:
        cols, cb = res[wname]
        with torch.no_grad():
            for p, c, s in zip(fm.cols, cols, fm.cscale): p.copy_(c.to(DEV) / s)
        fm.colbias = cb; base = evaluate(m, fm, D, D.dv)
        sh = rate_shift(m, fm, D); fm.colbias = [b + s for b, s in zip(cb, sh)]
        print(f"{tag} write {wname}: dev KL {base[0]:.4f} -> with reader rate law {evaluate(m, fm, D, D.dv)}", flush=True)
