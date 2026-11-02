"""Score B2 rulers on dev-5000, with the out-of-span share of their writes and the joint-span projection cost."""
import sys, numpy as np, torch
from inception import *
from family import Data
from w_fisher import real_kl, A_cols
from b2_train import FamilyRB
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"])
A = A_cols(m, fam); Aj = torch.cat(A, 0); sizes = [a.shape[0] for a in A]
builds = {}
for tag in sys.argv[1:]:
    ck = torch.load(f"cache/ckpt/{tag}.pt", map_location="cpu", weights_only=False)
    fm = FamilyRB(m, ck["fam"], ck["k"], ks=ck.get("ks", 5)); fm.add_rb(m); fm.load_state_dict(ck["state"]); fm = fm.to(DEV)
    cols = fm.weights()[2]; cb = [c.detach() for c in fm.colbias]; C = torch.cat(cols, 0)
    P = Aj @ torch.linalg.lstsq(Aj, C).solution
    print(f"{tag}: out-of-span share {np.round(((C - P).pow(2).sum(0) / C.pow(2).sum(0)).cpu().numpy(), 3)}; "
          f"reader-bias change mean {float(torch.cat(cb).mean()):+.4f} sd {float(torch.cat(cb).std()):.4f}", flush=True)
    builds[f"{tag} trained"] = (fm, cols, cb); builds[f"{tag} joint-span"] = (fm, list(torch.split(P, sizes, 0)), cb)
K = real_kl(m, fam, D, D.dv, builds)
for k, v in K.items(): print(f"  {k:28s} {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})")
for tag in sys.argv[1:]:
    d = K[f"{tag} joint-span"] - K[f"{tag} trained"]; print(f"  {tag} joint-span - trained {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})")
np.savez(f"cache/b2_score_{'_'.join(sys.argv[1:])}.npz", **{k.replace(' ', '_'): v for k, v in K.items()})
