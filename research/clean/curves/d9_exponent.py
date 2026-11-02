"""Read the trained write like a best-fit line: per reader, ratio trained/Bayes vs |rho|. Fit log ratio = a + b log|rho|
over readers whose signs agree and |rho| > .15 (where the ratio is defined). Per branch and all."""
import sys, numpy as np, torch
from inception import *
from family import Family, Data, IN4A
from d4_write import stats, writes
m = load_model(); D = Data(seed=0, ndev=100); fam = list(np.load("cache/s1_fine.npz")["fam"])
for tag in sys.argv[1:]:
    ck = torch.load(f"cache/ckpt/{tag}.pt", map_location="cpu", weights_only=False); fm = Family(m, fam, 1, ks=ck.get("ks", 5)).to(DEV); fm.load_state_dict(ck["state"])
    st = stats(m, fm, fam, D); res, g = writes(m, fam, st); allx, ally = [], []
    for j, name in enumerate(IN4A):
        mu, C = st[1] if j == 3 else st[0]
        A = getattr(m, name).weight.detach().cpu()[:, fam, 0, 0].double().numpy()
        covIc = A @ C[:10, 10]; varI = np.einsum("oi,ij,oj->o", A, C[:10, :10], A); rho = covIc / np.sqrt(varI * C[10, 10])
        t = fm.weights()[2][j][:, 0].cpu().numpy(); bay = res["bayes"][0][j][:, 0].numpy()
        ok = (np.sign(t) == np.sign(bay)) & (np.abs(rho) > .15)
        x, yv = np.log(np.abs(rho[ok])), np.log(t[ok] / bay[ok]); b, a = np.polyfit(x, yv, 1)
        print(f"{tag} {name:38s}: n {ok.sum():3d}/{len(t)}  log(trained/Bayes) = {a:+.2f} + {b:+.2f} log|rho|   (median ratio {np.exp(np.median(yv)):.2f}, median |rho| {np.median(np.abs(rho[ok])):.2f})")
        allx.append(x); ally.append(yv)
    x, yv = np.concatenate(allx), np.concatenate(ally); b, a = np.polyfit(x, yv, 1)
    print(f"{tag} ALL: log ratio = {a:+.2f} + {b:+.2f} log|rho|  corr {np.corrcoef(x, yv)[0,1]:.2f}  => trained ~ {np.exp(a):.2f} x Bayes x |rho|^{b:.2f}")
