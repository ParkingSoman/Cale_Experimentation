"""B1-B2 (NEURON-MATH Part VI): split training's local move S -> T (Gap B) into a shared re-routing W_F Delta-Lambda
(Prop. 11.1 form, least squares over all 368 readers), the reader-specific remainder, and the reader-bias change. B1:
dev-5000 KL of each partial build. B2: on the 500 x2b images, KL, L3 (exact second-order term, central difference) and
L3' (with the readers' gate flips) for S, T and the partial builds."""
import numpy as np, torch, copy
from circ import *

S = b3_train.FamilyRA(m, fam, 1); S.add_rb(m); S.add_ra(m); S.load_state_dict(torch.load("cache/x5f_factor_n2000.pt", map_location="cpu")); S = S.to(DEV)
T = b3_train.load_b3(m, "B3-k0-factorlocal")
cs = [float(c) for c in S.cscale]
dW = [(T.dW[j] - S.dW[j]).detach() * cs[j] for j in range(4)]                  # raw units
WFall = torch.cat(WF); dWall = torch.cat(dW)                                     # (368, 10), (368, 470)
dLam = torch.linalg.lstsq(WFall.cpu().double(), dWall.cpu().double()).solution.float().to(DEV)   # (10, 470)
span = [WF[j] @ dLam for j in range(4)]; rem = [dW[j] - span[j] for j in range(4)]
e = lambda L: float(sum((t ** 2).sum() for t in L))
print(f"|Delta-W|^2 {e(dW):.4f}: in-span {e(span):.4f} ({e(span) / e(dW):.2f}), remainder {e(rem):.4f} ({e(rem) / e(dW):.2f})", flush=True)
rr = torch.cat([(t ** 2).sum(1) for t in rem]).cpu().numpy(); np.save("cache/c5_rem_rows.npy", rr)
print("top-10 readers by remainder energy (global index over 1x1|3x3b|5x5b|pool = 192|96|16|64):", np.argsort(-rr)[:10].tolist(),
      f"share {np.sort(rr)[::-1][:10].sum() / rr.sum():.2f}; top-37 (10%) share {np.sort(rr)[::-1][:37].sum() / rr.sum():.2f}", flush=True)

def variant(parts, bias):
    fm = copy.deepcopy(S)
    with torch.no_grad():
        for j in range(4):
            fm.dW[j] += sum(P[j] for P in parts) / cs[j] if parts else 0
            if bias: fm.cbp[j].copy_(T.cbp[j])
    return fm

builds = {"S": S, "S+span": variant([span], False), "S+rem": variant([rem], False), "S+bias": variant([], True),
          "S+span+bias": variant([span], True), "T": T}
res = {}
for k, fm in builds.items():
    res[k] = per_image(fm); print(f"{k:12s} {res[k].mean():.4f} (se {res[k].std() / np.sqrt(len(res[k])):.4f})", flush=True)
for k in res:
    if k != "S": d = res[k] - res["S"]; print(f"  {k} minus S {d.mean():+.4f} (paired se {d.std() / np.sqrt(len(d)):.4f})")
np.savez("cache/c5_gapb.npz", **res)

# B2: second-order term on the 500 x2b images
D1 = Data(seed=0, ndev=1000); idx = np.sort(np.random.default_rng(3).choice(np.setdiff1d(D.dv, D1.dv), 500, replace=False))
convs = [getattr(m, n) for n in IN4A]
def reader_pre(x, p, fm=None):
    out = []
    for j, (c, inp) in enumerate(zip(convs, [x, x, x, p])):
        if fm is None: out.append(F.conv2d(inp, c.weight, c.bias)); continue
        W = (c.weight[:, fm.keep, 0, 0] + fm.dW[j] * fm.cscale[j])[:, :, None, None]
        out.append(F.conv2d(inp[:, fm.keep], W, c.bias + fm.colbias[j]))
    return out
def cont(post):
    a, b, cc, d = post
    return rest_from_4a(torch.cat([a, F.relu(C3(F.pad(b, (1, 1, 1, 1)))), F.relu(C5(F.pad(cc, (2, 2, 2, 2)))), d], 1))
acc = {k: dict(KL=[], L3=[], L3p=[]) for k in builds}
with torch.no_grad():
    for i in range(0, 500, 20):
        b = idx[i:i + 20]; x, p = family_inputs(b); a0 = reader_pre(x, p); Tt = torch.from_numpy(D.T[b].astype(np.float32)).to(DEV)
        pr = cont([F.relu(t) for t in a0]).softmax(1); var = lambda t: (pr * t ** 2).sum(1) - (pr * t).sum(1) ** 2
        for k, fm in builds.items():
            a1 = reader_pre(x, p, fm); E = [u - v for u, v in zip(a0, a1)]; Dl = [F.relu(u) - F.relu(v) for u, v in zip(a0, a1)]
            acc[k]["KL"].append(kl(Tt, cont([F.relu(v) for v in a1])).cpu().numpy())
            eps = 1e-2
            jp = (cont([F.relu(t + eps * u) for t, u in zip(a0, E)]) - cont([F.relu(t - eps * u) for t, u in zip(a0, E)])) / (2 * eps)
            jq = (cont([F.relu(t) + eps * u for t, u in zip(a0, Dl)]) - cont([F.relu(t) - eps * u for t, u in zip(a0, Dl)])) / (2 * eps)
            acc[k]["L3"].append((0.5 * var(jp)).cpu().numpy()); acc[k]["L3p"].append((0.5 * var(jq)).cpu().numpy())
r2 = {k: {q: np.concatenate(v) for q, v in d.items()} for k, d in acc.items()}
np.savez("cache/c5_b2.npz", **{f"{k}_{q}": v for k, d in r2.items() for q, v in d.items()})
print(f"{'build':12s} {'KL':>8s} {'L3':>8s} {'L3p':>8s}  (500 x2b images)")
for k, d in r2.items(): print(f"{k:12s} {d['KL'].mean():8.4f} {d['L3'].mean():8.4f} {d['L3p'].mean():8.4f}")
for k in r2:
    if k == "S": continue
    for q in ["KL", "L3", "L3p"]:
        d = r2[k][q] - r2["S"][q]; print(f"  {k} minus S, {q}: {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})")
