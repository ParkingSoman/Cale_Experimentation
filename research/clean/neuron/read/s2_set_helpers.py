"""Helpers copied from s2_set.py (b_plus, t_minus, test) for reuse; call init() first."""

import sys, os, json, time, copy
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcommon import *
t0 = time.time(); say = lambda *a: print(*a, f"[{time.time()-t0:.0f}s]", flush=True)
base = mB = mT = Ss = calib = P = None
def init(n, p):
    global base, mB, mT, Ss, calib, P, pos
    base, mB, mT, Ss, calib = n; P = p
    pos = [{int(b): k for k, b in enumerate(S.tolist())} for S in Ss]

def b_plus(surv, dele):
    """B with survivors (g, base index) from T, deleted (g, base index) re-added; blocks holding re-added neurons are rebuilt
    from Base with S + extra (writes re-solved), then the listed survivors there get T's numbers.
    Returns (network, reference): the reference is B with the same blocks rebuilt from S alone (the null)."""
    m, ref = copy.deepcopy(mB), copy.deepcopy(mB)
    extra = {}
    for g, j in dele: extra.setdefault(g, []).append(j)
    for g, js in extra.items():
        rebuild_block(ref, base, g, Ss[g], calib)
        rebuild_block(m, base, g, torch.cat([Ss[g], torch.tensor(js)]), calib)
    with torch.no_grad():
        for g, b in surv:
            if g in extra:
                newS = sorted(Ss[g].tolist() + extra[g]); k = newS.index(b); j = pos[g][b]
                L1(m, g).weight[k].copy_(L1(mT, g).weight[j]); L1(m, g).bias[k].copy_(L1(mT, g).bias[j]); L2(m, g).weight[:, k].copy_(L2(mT, g).weight[:, j])
            else:
                set_neurons(m, mT, [(g, pos[g][b])])
    return m, ref

def t_minus(surv):
    m = copy.deepcopy(mT); set_neurons(m, mB, [(g, pos[g][b]) for g, b in surv]); return m

def test(surv, dele, rows):
    kB, kT = rows.kl(mB), rows.kl(mT); gap = (kB - kT).mean()
    m, ref = b_plus(surv, dele); gain = rows.kl(ref) - rows.kl(m)
    cost = rows.kl(t_minus(surv)) - kT if surv else np.zeros_like(kT)
    return dict(gap=gap, a_cost=cost.mean() / gap, a_ci=(np.array(boot_ci(cost)) / gap).tolist(),
                b_gain=gain.mean() / gap, b_ci=(np.array(boot_ci(gain)) / gap).tolist())

