"""Phase 4 cold test: ONE run on the sealed split (10 images per class, 10k). KL to intact logits and top-1.
usage: cold.py (writes results/curves-cold.json)"""
import json, numpy as np, torch
from inception import *
from family import Family, Data, evaluate, IN4A
from d4_write import stats, writes
from derive import derived_read, bayes_write

m = load_model(); D = Data(seed=0, ndev=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); S = D.se
out = {}; per = {}
from family import forward_family, kl as klf
def evaluate_full(fm, idx, bs=100):
    ks, tp = [], []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]; lg = forward_family(m, fm, inputs_3b(m, to_3a(m, prep(D.X[b]))))
            t = torch.from_numpy(D.T[b].astype(np.float32)).to(DEV)
            ks.append(klf(t, lg).cpu().numpy()); tp.append(lg[:, D.lab].argmax(1).cpu().numpy() == D.y[b])
    k = np.concatenate(ks); return k, float(np.concatenate(tp).mean())
def run(name, fm):
    k, top = evaluate_full(fm, S); per[name] = k; kl_, se_ = float(k.mean()), float(k.std() / np.sqrt(len(k)))
    out[name] = dict(kl=kl_, se=se_, top1=top)
    print(f"{name:60s} sealed KL {kl_:.4f} (se {se_:.4f})  top-1 {top:.4f}", flush=True)
def load(tag):
    ck = torch.load(f"cache/ckpt/{tag}.pt", map_location="cpu", weights_only=False)
    fm = Family(m, ck["fam"], ck["k"], ks=ck.get("ks", 5)).to(DEV); fm.load_state_dict(ck["state"]); return fm

intact_top = float((D.T[S][:, D.lab].argmax(1) == D.y[S]).mean()); out["intact"] = dict(kl=0.0, top1=intact_top); print("intact top-1", intact_top)
# removal (zero columns) and constant-only (zero columns + reader-side mean of the family)
fm = Family(m, fam, 1).to(DEV)
with torch.no_grad():
    for p in fm.cols: p.zero_()
fm.bias.data.fill_(-1e3)
run("family removed (zero)", fm)
st = stats(m, fm, fam, D)
cb = []
for j, name in enumerate(IN4A):
    mu = (st[1] if j == 3 else st[0])[0]; A = getattr(m, name).weight.detach().cpu()[:, fam, 0, 0].double().numpy()
    cb.append(torch.tensor(A @ mu[:10], dtype=torch.float32).to(DEV))
fm.colbias = cb; run("constant only (reader-side family mean)", fm)
# simple cut: keep the member with the largest variance (train), others -> reader-side mean
var = np.diag(st[0][1])[:10]; keep = int(np.argmax(var)); kn = fam[keep]
fm = Family(m, fam, 1, init=f"member:{kn}").to(DEV); cbk = []
for j, name in enumerate(IN4A):
    mu = (st[1] if j == 3 else st[0])[0]; A = getattr(m, name).weight.detach().cpu()[:, fam, 0, 0].double().numpy()
    others = [i for i in range(10) if i != keep]; cbk.append(torch.tensor(A[:, others] @ mu[others], dtype=torch.float32).to(DEV))
fm.colbias = cbk; run(f"simple cut: keep member {kn} (largest variance), others -> mean", fm)
# D1 pre-registered: moved kernel sum r*=4, sum of biases, Bayes write
fm = Family(m, fam, 1, ks=13).to(DEV); K, b = derived_read(m, fam, 4); _, _, c0 = fm.weights()
fm.set_weights(torch.tensor(K)[None].to(DEV), torch.tensor([b]).to(DEV), [c.to(DEV) for c in c0])
cols, cbias, _, _ = bayes_write(m, fm, fam, D)
fm.set_weights(torch.tensor(K)[None].to(DEV), torch.tensor([b]).to(DEV), [c.to(DEV) for c in cols]); fm.colbias = cbias
run("D1 PRE-REGISTERED derived (moved kernel sum r*=4, Bayes write)", fm)
# derived reads with the per-reader variance write
for tag, lab in [("derived2-r4-ks13", "D2 projection r*=4 13x13 + vmr write (r pre-registered)"),
                 ("derived2-r1-ks7", "D2 projection r=1 7x7 + vmr write (best of 9 on dev)")]:
    fm = load(tag); res, g = writes(m, fam, stats(m, fm, fam, D)); cols, cbv = res["vmr"]
    with torch.no_grad():
        for p, c, s in zip(fm.cols, cols, fm.cscale): p.copy_(c.to(DEV) / s)
    fm.colbias = cbv; run(lab, fm)
# loop neuron with the Bayes write too
fm = load("derived2-r1-ks7"); res, g = writes(m, fam, stats(m, fm, fam, D)); cols, cbv = res["bayes"]
with torch.no_grad():
    for p, c, s_ in zip(fm.cols, cols, fm.cscale): p.copy_(c.to(DEV) / s_)
fm.colbias = cbv; run("D2 loop neuron r=1 7x7 + Bayes write (r chosen on dev)", fm)
# derived k-curve: keep evenly spaced members + Bayes / vmk estimate of the removed ones
from d12_k3 import member_stats, build
mst = member_stats(m, fam, D)
var = np.diag(mst[0][1]); naive = [fam[i] for i in np.argsort(-var)[:3]]
for keep, lab in [([330], "k=1 keep 330"), ([342], "k=1 keep 342"), ([330, 342], "k=2 keep 10/180 deg"),
                  ([330, 388, 385], "k=3 keep 10/120/250 deg (first pick, made for the training run)"),
                  ([340, 343, 406], "k=3 keep 50/170/290 (control, also even)"), ([330, 340, 324], "k=3 keep 10/50/70 (control, adjacent)"),
                  (naive, f"k=3 NAIVE: 3 largest-variance members {naive}"),
                  ([330, 324, 343, 385, 406], "k=5 keep 10/70/170/250/290")]:
    fmk = Family(m, fam, len(keep), init="member:" + ",".join(map(str, keep))).to(DEV)
    for mode in (["const", "bayes", "vmk"] if len(keep) >= 3 else ["const", "bayes"]):
        cols, cbs = build(m, fam, keep, mst, mode)
        with torch.no_grad():
            for p, c, s_ in zip(fmk.cols, cols, fmk.cscale): p.copy_(c.to(DEV) / s_)
        fmk.colbias = cbs; run(f"derived {lab} + {mode}", fmk)
# k=3 read frozen at members + trained write (measuring tool)
try:
    ck = torch.load("cache/ckpt/k3-writeonly-330-388-385.pt", map_location="cpu", weights_only=False)
    fmk = Family(m, fam, 3, init="member:330,388,385").to(DEV); st_ = {k_: v for k_, v in ck["state"].items() if not k_.startswith("cbp")}
    fmk.load_state_dict(st_, strict=False); fmk.colbias = [b.to(DEV) * 100.0 for b in ck["cbp"]]
    run("k=3 members 10/120/250, write trained only (measuring tool)", fmk)
except FileNotFoundError: print("missing k3 write-only")
# trained references
for tag, lab in [("L-k1-m379-lr3e-2", "trained k=1 (start 379)"), ("L-k1-r-lr3e-2", "trained k=1 (random start)"),
                 ("L-k3-r-lr3e-2", "trained k=3 (random start)"), ("L-k3-m-lr3e-2", "trained k=3 (start 3 members)"),
                 ("L-k1-ks13-r-lr3e-2", "trained k=1 13x13 window"), ("L-k10-r-lr3e-2", "trained k=10 random (harness)")]:
    try: run(lab, load(tag))
    except FileNotFoundError: print("missing", tag)
def paired(a, b):
    ka = [n for n in per if all(x in n for x in a.split("&&"))]; kb = [n for n in per if all(x in n for x in b.split("&&"))]
    if not ka or not kb: return
    d = per[ka[0]] - per[kb[0]]; out[f"paired: {ka[0]} MINUS {kb[0]}"] = dict(diff=float(d.mean()), se=float(d.std() / np.sqrt(len(d))))
    print(f"PAIRED  [{ka[0]}]  -  [{kb[0]}]  = {d.mean():+.4f} (se {d.std()/np.sqrt(len(d)):.4f})", flush=True)
paired("derived k=3 keep 10/120/250 deg (first pick, made for the training run) + bayes", "trained k=3 (random start)")
paired("derived k=3 keep 10/120/250 deg (first pick, made for the training run) + bayes", "trained k=1 (random start)")
paired("derived k=3 keep 10/120/250 deg (first pick, made for the training run) + bayes", "derived k=3 keep 50/170/290 (control, also even) + bayes")
paired("derived k=3 keep 10/120/250 deg (first pick, made for the training run) + bayes", "derived k=3 NAIVE&&+ bayes")
paired("derived k=3 keep 10/120/250 deg (first pick, made for the training run) + bayes", "derived k=3 keep 10/120/250 deg (first pick, made for the training run) + vmk")
paired("D2 projection r=1 7x7 + vmr write", "derived k=1 keep 342 + bayes")
paired("D2 projection r=1 7x7 + vmr write", "trained k=1 (random start)")
paired("trained k=3 (start 3 members)", "derived k=3 keep 10/120/250 deg (first pick, made for the training run) + bayes")
np.savez("cache/cold_per_image.npz", **{k_.replace(" ", "_")[:120]: v for k_, v in per.items()})
json.dump(out, open("../../../results/curves-cold.json", "w"), indent=1)
