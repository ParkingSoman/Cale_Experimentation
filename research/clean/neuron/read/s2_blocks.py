"""N10 step 2a: checks, then whole-block worth for birds in both orders on the choose set.
Checks: which parameters T changed relative to B; null rebuild (block g rebuilt from Base with S = B's block)."""
import sys, os, json, time, copy
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcommon import *
t0 = time.time()
base, mB, mT, Ss, calib = nets(); P = photo_sets()
sB, sT = mB.state_dict(), mT.state_dict()
changed = [k for k in sB if not torch.equal(sB[k], sT[k])]
kinds = sorted({k.split(".")[-2] + "." + k.split(".")[-1] if "block" in k else k for k in changed})
print("T changed", len(changed), "tensors; kinds:", kinds[:12], flush=True)
# null rebuild
m = copy.deepcopy(mB); worst = 0
for g in (0, 14, 28):
    rebuild_block(m, base, g, Ss[g], calib)
    worst = max(worst, float((L1(m, g).weight - L1(mB, g).weight).abs().max()), float((L2(m, g).weight - L2(mB, g).weight).abs().max()), float((L2(m, g).bias - L2(mB, g).bias).abs().max()))
print(f"null rebuild max weight diff {worst:.2e}", flush=True)
res = dict(changed=changed)
for nm, rows in (("choose", P["choose"]), ("score", P["score"])):
    kB, kT = rows.kl(mB), rows.kl(mT)
    res[nm] = dict(n=len(kB), B=kB.mean(), T=kT.mean(), gap=(kB - kT).mean(), gap_ci=boot_ci(kB - kT))
    print(nm, {k: v for k, v in res[nm].items()}, f"{time.time()-t0:.0f}s", flush=True)
    if nm == "choose": kB_c, kT_c = kB, kT
kl0 = P["choose"].kl(m); print(f"null rebuild KL change on choose {np.abs(kl0 - kB_c).max():.2e}", flush=True)
res["null_rebuild_max_kl_change"] = float(np.abs(kl0 - kB_c).max())
gap = res["choose"]["gap"]
blocks = {}
for g in range(30):
    m = copy.deepcopy(mT); set_block(m, mB, [g]); cost = P["choose"].kl(m) - kT_c
    m = copy.deepcopy(mB); set_block(m, mT, [g]); gain = kB_c - P["choose"].kl(m)
    blocks[g] = dict(T_cost=cost.mean(), T_cost_ci=boot_ci(cost), B_gain=gain.mean(), B_gain_ci=boot_ci(gain))
    print(f"block {g:2d}: T minus block cost {cost.mean():+.4f} ({cost.mean()/gap:+.3f} of gap)  B plus block gain {gain.mean():+.4f} ({gain.mean()/gap:+.3f})  {time.time()-t0:.0f}s", flush=True)
    if g == 0 and "--smoke" in sys.argv: sys.exit()
res["blocks"] = blocks
# cumulative, T order, largest cost first
order = sorted(range(30), key=lambda g: -blocks[g]["T_cost"]); m = copy.deepcopy(mT); cum = []
for k, g in enumerate(order):
    set_block(m, mB, [g]); c = (P["choose"].kl(m) - kT_c).mean(); cum.append((g, c / gap))
    print(f"T minus top {k+1} blocks (+{g}): cost {c/gap:.3f} of gap", flush=True)
    if c / gap >= 0.5 and k >= 4: break
res["T_cumulative"] = cum
json.dump(conv(res), open(os.path.join(OUT, "s2_blocks.json"), "w"), indent=1)
