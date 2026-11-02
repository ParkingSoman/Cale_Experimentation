"""C.3': local window model of the per-image metric. L_a(B) = 1/2 E_x E_x' sum_c (sum_{j,p} gt_c(x')(j,p) a_x(p) e_B(x)(j,p))^2
with gt = G(x') divided by a_x'(p) (the metric image's own oracle window); a_x from the oracle, the family signal, or the
reader outputs."""
import numpy as np, torch
src = open("c11_exactavg.py").read(); exec(src.split("Lf = {k: 0.0")[0])
Gr = G.reshape(500, 10, 368, 196).float()
wo = Gr.pow(2).sum((1, 2)).sqrt()                                                  # (500, 196) oracle window of metric images
Gt = (Gr / wo.clamp(min=1e-6)[:, None, None]).half()
E, A = {}, {"o": [], "h": [], "r": []}
for i in range(0, 500, 20):
    b = Xa[i:i + 20]; x, p = family_inputs(b); Gi = reader_grads(b).reshape(len(b), 10, 368, 196)
    with torch.no_grad():
        a0 = pre(x, p)
        for k, fm in builds.items(): E.setdefault(k, []).append((a0 - pre(x, p, fm)).flatten(2).half())
        A["o"].append(Gi.pow(2).sum((1, 2)).sqrt()); A["h"].append(x[:, fam].flatten(2).norm(dim=1))
        A["r"].append(F.relu(a0).flatten(2).norm(dim=1))
E = {k: torch.cat(v).float() for k, v in E.items()}; A = {k: torch.cat(v) for k, v in A.items()}
for wk in ["o", "h", "r"]:
    a = A[wk] / A[wk].mean(); Gm = Gt.float().reshape(5000, -1)
    vals = {}
    for k in builds:
        ea = (E[k] * a[:, None, :]).reshape(500, -1)
        vals[k] = float(((Gm @ ea.T) ** 2).sum()) / 500 / 500 / 2
    print(f"window {wk}: " + "  ".join(f"{k} {v:.4f}" for k, v in vals.items()) + f"   S+rem/S {vals['S+rem'] / vals['S']:.3f}", flush=True)
cc = lambda u, v: float(np.corrcoef(u.flatten().cpu().numpy(), v.flatten().cpu().numpy())[0, 1])
print(f"corr of windows with the oracle: family signal {cc(A['h'], A['o']):.3f}, reader outputs {cc(A['r'], A['o']):.3f}")
