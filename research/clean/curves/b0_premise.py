"""B0 (premise): does the curve family see ANIMALS (classes 0-397) and REST (398-999) differently, beyond split-half
noise? Train split only, 1750 images per half, two halves per mix. Prediction in CURVES-LOG.md (MIX, B0)."""
import numpy as np, torch
from inception import *
from moments import compile_read, read_from, LO, HI
m = load_model(); X, y = load_images(); tr, dv, se = split(y); fam = list(np.load("cache/s1_fine.npz")["fam"])
rng = np.random.default_rng(23); halves = {}
for name, sel in [("ani", y[tr] < 398), ("rest", y[tr] >= 398)]:
    ix = rng.permutation(tr[sel])[:3500]; halves[name + "A"], halves[name + "B"] = ix[:1750], ix[1750:]
M = {k: compile_read(m, X, fam, v) for k, v in halves.items()}
def stats(P):
    on = (P["on"] / P["W"]).numpy(); mu = (P["mu"] / P["W"]).numpy(); S2 = (P["S2"] / P["W"]).numpy()
    C = S2 - np.outer(mu, mu); sd = np.sqrt(np.diag(C)); R = C / np.outer(sd, sd)
    return on, mu, R[np.triu_indices(10, 1)]
st = {k: stats(P) for k, P in M.items()}; rd = {k: read_from(P) for k, P in M.items()}
ev = np.sort(np.random.default_rng(5).choice(dv, 200, replace=False)); pre = {}
with torch.no_grad():
    ins = [inputs_3b(m, to_3a(m, prep(X[ev[i:i + 50]])))[2] for i in range(0, 200, 50)]
    for k, (K, b) in rd.items():
        p = K.shape[-1] // 2
        pre[k] = torch.cat([F.conv2d(F.pad(a, (p, p, p, p)), K[None].to(DEV), torch.tensor([b], device=DEV))[:, 0, LO:HI, LO:HI] for a in ins]).flatten().cpu().numpy()
within = [("aniA", "aniB"), ("restA", "restB")]; between = [(a, b) for a in ["aniA", "aniB"] for b in ["restA", "restB"]]
def rep(f, label):
    w = np.mean([f(a, b) for a, b in within]); bt = np.mean([f(a, b) for a, b in between])
    print(f"{label:34s} within {w:.4f}  between {bt:.4f}  ratio {bt / max(w, 1e-12):.2f}", flush=True)
rep(lambda a, b: np.abs(st[a][0] - st[b][0]).mean(), "fire rate |diff| (mean over 10)")
rep(lambda a, b: np.abs(st[a][1] - st[b][1]).mean() / st[a][1].mean(), "mean activation |diff| (relative)")
rep(lambda a, b: np.abs(st[a][2] - st[b][2]).mean(), "co-activation corr |diff| (45)")
rep(lambda a, b: 1 - float(torch.nn.functional.cosine_similarity(rd[a][0].flatten(), rd[b][0].flatten(), 0)), "read kernel 1 - cos")
rep(lambda a, b: 1 - np.corrcoef(pre[a], pre[b])[0, 1], "read pre-activation 1 - corr (dev)")
for k in M: print(k, "rate", st[k][0].round(3), "| read bias %.0f" % rd[k][1])
torch.save(dict(M=M, halves=halves), "/private/tmp/claude-501/-Users-shonusengupta-Cale-Experimentation/c4ebeb4f-1878-49b9-8b30-81ce79690a1a/scratchpad/b0_moments.pt")
