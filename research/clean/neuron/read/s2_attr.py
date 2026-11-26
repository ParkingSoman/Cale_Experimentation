"""N10 step 2b: attribution shortlist for birds, then real patching (both orders), set by the fixed rule, collective test.
--smoke: 60 choose photos, 3 items of each kind, no collective test."""
import sys, os, json, time, copy
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcommon import *
SMOKE = "--smoke" in sys.argv
t0 = time.time(); say = lambda *a: print(*a, f"[{time.time()-t0:.0f}s]", flush=True)
base, mB, mT, Ss, calib = nets(); P = photo_sets()
CH = P["choose"]
if SMOKE: CH = Rows([(CH.parts[0][0], CH.parts[0][1][:60])])
NBL = 30

def batches(rows, bs=25):
    for c, r in rows.parts:
        for a in range(0, len(r), bs):
            yield c.get(r[a:a + bs])

# ---------- attribution ----------
def grads(m, capture=False):
    """Mean-over-photos KL gradient wrt every block's W1, b1, W2, b2. If capture: also the deleted-neuron scores (B order)."""
    for p_ in m.parameters(): p_.requires_grad_(False)
    params = [(L1(m, g), L2(m, g)) for g in range(NBL)]
    for a, b in params:
        for t in (a.weight, a.bias, b.weight, b.bias): t.requires_grad_(True); t.grad = None
    n = len(CH); dscore = [None] * NBL
    if capture:
        pre = []
        for g in range(NBL):
            mg, K = calib[g]; S = Ss[g]; h = K.shape[0]
            D = torch.from_numpy(np.setdiff1d(np.arange(h), S.numpy()))
            ridge = 1e-6 * K.diagonal().mean()
            KSS = K[S][:, S] + ridge * torch.eye(len(S), dtype=torch.float64)
            Z = torch.linalg.solve(KSS, K[S][:, D])                        # (k, |D|)
            Rc = K[:, D] - K[:, S] @ Z                                     # cov(all neurons, residual of each j)
            v = K[D, D] - (K[D][:, S] * Z.T).sum(1)
            ls, Wb, _ = write_mat(C.blocks(base)[g])
            beta = (Wb @ Rc) / (v.clamp_min(0) + ridge)                          # (d, |D|): output change per unit residual
            bw = L1(base, g)
            pre.append(dict(D=D, Z=Z.float().to(C.DEV), beta=beta.float().to(C.DEV), mS=mg[S].float().to(C.DEV), mD=mg[D].float().to(C.DEV),
                            W1D=bw.weight[D.to(C.DEV)].detach(), b1D=bw.bias[D.to(C.DEV)].detach(), ls=ls.float().to(C.DEV)))
            dscore[g] = torch.zeros(len(D), device=C.DEV)
        store = {}
        hs = []
        for g, blk in enumerate(C.blocks(m)):
            hs.append(blk.block[3].register_forward_hook(lambda mod, i, o, g=g: store.__setitem__(("u", g), i[0].detach())))
            hs.append(blk.block[4].register_forward_hook(lambda mod, i, o, g=g: store.__setitem__(("gS", g), o.detach())))
            def f5(mod, i, o, g=g):
                o.retain_grad(); store[("y", g)] = o
            hs.append(blk.block[5].register_forward_hook(f5))
    for x, z in batches(CH):
        with torch.enable_grad():
            y = C.tail(m, x); loss = C.kl_rows(z, y).sum() / n; loss.backward()
        if capture:
            with torch.no_grad():
                for g in range(NBL):
                    q = pre[g]; u = store[("u", g)]
                    gD = F.gelu(u @ q["W1D"].T + q["b1D"])
                    r = (gD - q["mD"]) - (store[("gS", g)] - q["mS"]) @ q["Z"]
                    gy = store[("y", g)].grad / q["ls"]                    # grad wrt the post-layer-scale output
                    A = gy.reshape(-1, gy.shape[-1]) @ q["beta"]
                    dscore[g] += (A * r.reshape(-1, r.shape[-1])).sum(0)
            store.clear()
    if capture:
        for h_ in hs: h_.remove()
    out = [(a.weight.grad.clone(), a.bias.grad.clone(), b.weight.grad.clone(), b.bias.grad.clone()) for a, b in params]
    for a, b in params:
        for t in (a.weight, a.bias, b.weight, b.bias): t.requires_grad_(False); t.grad = None
    return out, ([(pre[g]["D"], -dscore[g].cpu()) for g in range(NBL)] if capture else None)

from common import write_mat
gB, dele = grads(mB, capture=True); say("grads B")
gT, _ = grads(mT); say("grads T")
surv = []                                   # (pred_gain_B, pred_cost_T, g, j)
for g in range(NBL):
    a1, a2, b1_, b2_ = L1(mB, g), L2(mB, g), L1(mT, g), L2(mT, g)
    dW1, db1, dW2 = b1_.weight - a1.weight, b1_.bias - a1.bias, b2_.weight - a2.weight
    pg = -((gB[g][0] * dW1).sum(1) + gB[g][1] * db1 + (gB[g][2] * dW2).sum(0))
    pc = -((gT[g][0] * dW1).sum(1) + gT[g][1] * db1 + (gT[g][2] * dW2).sum(0))   # cost of moving T toward B (sign fixed after the run)
    for j in range(len(pg)): surv.append((float(pg[j]), float(pc[j]), g, j))
dels = [(float(s[i]), g, int(D[i])) for g, (D, s) in enumerate(dele) for i in range(len(D))]
say(f"survivors {len(surv)}, deleted {len(dels)}; top pred gains surv {sorted(s[0] for s in surv)[-3:]}, del {sorted(d[0] for d in dels)[-3:]}")

# ---------- real patching ----------
kB, kT = CH.kl(mB), CH.kl(mT); gap = (kB - kT).mean(); say(f"choose gap {gap:.4f}")
NS, NR, ND = (3, 3, 3) if SMOKE else (40, 30, 40)
top_s = sorted(surv, key=lambda t: -t[0])[:NS]
rng = np.random.default_rng(0); rest = [s for s in surv if s not in top_s]
rand_s = [rest[i] for i in rng.choice(len(rest), NR, replace=False)]
top_d = sorted(dels, key=lambda t: -t[0])[:ND]
res = dict(gap=gap, surv=[], rand=[], dele=[], wbias=[])
for tag, lst in (("surv", top_s), ("rand", rand_s)):
    for pgain, pcost, g, j in lst:
        m = copy.deepcopy(mB); set_neurons(m, mT, [(g, j)]); gain = kB - CH.kl(m)
        m = copy.deepcopy(mT); set_neurons(m, mB, [(g, j)]); cost = CH.kl(m) - kT
        res[tag].append(dict(g=g, j=j, base_index=int(Ss[g][j]), pred_gain=pgain, pred_cost=pcost, gain=gain.mean(), gain_ci=boot_ci(gain), cost=cost.mean(), cost_ci=boot_ci(cost)))
        say(f"{tag} g{g} j{j}: pred gain {pgain:+.5f} real {gain.mean():+.5f} | pred cost {pcost:+.5f} real {cost.mean():+.5f}")
null = {}
for pgain, g, j in top_d:
    if g not in null:
        m = copy.deepcopy(mB); rebuild_block(m, base, g, Ss[g], calib); null[g] = CH.kl(m)
    m = copy.deepcopy(mB); rebuild_block(m, base, g, torch.cat([Ss[g], torch.tensor([j])]), calib); gain = null[g] - CH.kl(m)
    res["dele"].append(dict(g=g, base_index=j, pred_gain=pgain, gain=gain.mean(), gain_ci=boot_ci(gain)))
    say(f"deleted g{g} n{j}: pred gain {pgain:+.5f} real {gain.mean():+.5f}")
if not SMOKE:
    for g in range(NBL):
        m = copy.deepcopy(mB); set_wbias(m, mT, [g]); gain = (kB - CH.kl(m)).mean()
        m = copy.deepcopy(mT); set_wbias(m, mB, [g]); cost = (CH.kl(m) - kT).mean()
        res["wbias"].append(dict(g=g, gain=gain, cost=cost))
    say("write biases: gain sum %.4f, cost sum %.4f" % (sum(w["gain"] for w in res["wbias"]), sum(w["cost"] for w in res["wbias"])))
json.dump(conv(res), open(os.path.join(OUT, "s2_attr_smoke.json" if SMOKE else "s2_attr.json"), "w"), indent=1)
torch.save(dict(surv=surv, dels=dels), os.path.join(OUT, "s2_attr_all.pt" if not SMOKE else "s2_attr_all_smoke.pt"))
say("done")
