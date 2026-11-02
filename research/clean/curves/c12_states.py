"""C.3 (a): a two-state (circuit on / off) averaged objective. States from the per-image second-order cost of S."""
import numpy as np, torch
src = open("c11_exactavg.py").read(); exec(src.split("Lf = {k: 0.0")[0])
def per_image_cost(imgs, fm):
    out = []
    for i in range(0, len(imgs), 20):
        b = imgs[i:i + 20]; x, p = family_inputs(b); Gi = reader_grads(b).flatten(1)
        with torch.no_grad(): e = (pre(x, p) - pre(x, p, fm)).flatten(1); out.append((((Gi.reshape(len(b), 10, -1) * e[:, None]).sum(2)) ** 2).sum(1).cpu())
    return torch.cat(out).numpy()
cm, ca = per_image_cost(Xm, S), per_image_cost(Xa, S)
E = {}
for i in range(0, 500, 20):
    b = Xa[i:i + 20]; x, p = family_inputs(b)
    with torch.no_grad():
        a0 = pre(x, p)
        for k, fm in builds.items(): E.setdefault(k, []).append((a0 - pre(x, p, fm)).flatten(1).half())
E = {k: torch.cat(v) for k, v in E.items()}
Gr = G.reshape(500, 10, -1)
for q in [.95, .90]:
    sm, sa = cm >= np.quantile(cm, q), ca >= np.quantile(ca, q)
    print(f"top {1 - q:.0%}: metric-image cost share {cm[sm].sum() / cm.sum():.2f}, activity-image share {ca[sa].sum() / ca.sum():.2f}", flush=True)
    for k in builds:
        tot = 0.0; one = 0.0
        for st in [True, False]:
            Gs = Gr[torch.from_numpy(sm == st)].flatten(0, 1).float(); Es = E[k][torch.from_numpy(sa == st)].float()
            pi = float((sa == st).mean())
            tot += pi * float(((Gs @ Es.T) ** 2).sum()) / (Gs.shape[0] / 10) / Es.shape[0] / 2
        print(f"   {k:10s} L_2s {tot:.4f}", flush=True)
