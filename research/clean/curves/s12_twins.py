"""Two trained coarse neurons (same k, different start): are they one object? Kernel cosine, activation correlation
on natural images, firing rate, loop-sum correlation (r=1..4), write columns vs mean original column per branch.
usage: s12_twins.py tagA tagB [more tags...]"""
import sys, numpy as np, torch
from inception import *
from family import Family, IN4A
from s7_lib import moved
tags = sys.argv[1:]
m = load_model(); X, y = load_images(); tr, dv, se = split(y); fam = list(np.load("cache/s1_fine.npz")["fam"])
fms = []
for t in tags:
    ck = torch.load(f"cache/ckpt/{t}.pt", map_location="cpu", weights_only=False); fm = Family(m, ck["fam"], ck["k"], ks=ck.get("ks", 5)).to(DEV)
    fm.load_state_dict(ck["state"]); fms.append(fm)
idx = np.sort(np.random.default_rng(3).choice(tr, 300, replace=False)); Zs = []; Cs = [[] for _ in tags]
with torch.no_grad():
    for i in range(0, len(idx), 100):
        ins = inputs_3b(m, to_3a(m, prep(X[idx[i:i+100]])))
        Zs.append(F.relu(pre_3b(m, ins[0])[:, fam]).cpu().numpy())
        for j, fm in enumerate(fms): Cs[j].append(fm.pre(ins)[:, 0].cpu().numpy())
Z = np.concatenate(Zs); C = [np.concatenate(c) for c in Cs]; M = (slice(None), slice(4, 24), slice(4, 24))
np.set_printoptions(precision=3, suppress=True, linewidth=160)
for j, t in enumerate(tags):
    c = C[j][M].ravel(); fm = fms[j]
    lo = [np.corrcoef(c, moved(Z, r).sum(1)[M].ravel())[0, 1] for r in [0, 1, 2, 3, 4]]
    print(f"{t}: firing rate {(c > 0).mean():.3f}  corr with loop sum r=0..4: {np.round(lo, 3)}  bias {fm.weights()[1].item():.0f}")
    for nm, col in zip(IN4A, fm.weights()[2]):
        C0 = getattr(m, nm).weight.detach().cpu()[:, fam, 0, 0].numpy(); mc = C0.mean(1); cv = col[:, 0].cpu().numpy()
        print(f"   {nm:38s} cos(col, mean original col) {cv @ mc / np.linalg.norm(cv) / np.linalg.norm(mc):+.3f}   |col| / |mean col| {np.linalg.norm(cv)/np.linalg.norm(mc):.2f}")
for a in range(len(tags)):
    for b in range(a + 1, len(tags)):
        ka, kb = fms[a].weights()[0][0].cpu().numpy(), fms[b].weights()[0][0].cpu().numpy()
        if ka.shape == kb.shape: kc = (ka * kb).sum() / np.linalg.norm(ka) / np.linalg.norm(kb)
        else: kc = float("nan")
        print(f"{tags[a]} vs {tags[b]}: kernel cos {kc:+.3f}  activation corr {np.corrcoef(C[a][M].ravel(), C[b][M].ravel())[0,1]:.3f}  "
              f"post-ReLU corr {np.corrcoef(np.maximum(C[a][M].ravel(),0), np.maximum(C[b][M].ravel(),0))[0,1]:.3f}")
