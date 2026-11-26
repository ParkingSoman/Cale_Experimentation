"""N10 pilot: capture pooled values of a few neurons (and their feeders and readers) for reading.
Input: results/read/pilot_neurons.json = [{"name": "n1", "g": block, "base_index": i}, ...] (chosen by measured worth).
Base: every reading photo (35,000; Base never saw our split). B and T: only photos T never trained on
(held, test, and the unused odd-position photos), so T's memorized fit photos never enter a reading.
Per photo and neuron: max and min over positions of the PRE-activation (both poles), and where they occur.
A deleted neuron has no value in B or T; there its 'virtual' value = Base's read row applied to B's / T's block input
(does the information it read still reach that block?).
Feeders (block g-1) and readers (block g+1): top 5 each by an approximate direct edge,
edge(i -> j) = sum_d W1_next[j, d] * gamma_d * kappa_d * write_i[d], gamma = LayerNorm weight, kappa = sum of the dwconv kernel
(ignores the LayerNorm scale and spatial mixing). Base only."""
import sys, os, json, time, copy
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rcommon import *
t0 = time.time(); say = lambda *a: print(*a, f"[{time.time()-t0:.0f}s]", flush=True)
NE = json.load(open(os.path.join(OUT, "pilot_neurons.json")))
base, mB, mT, Ss, calib = nets()
pos = [{int(b): k for k, b in enumerate(S.tolist())} for S in Ss]
BL = C.blocks(base)
def edges_out(g, i):
    if g + 1 >= 30: return []
    nb = BL[g + 1]; w = C.write_mat(BL[g])[1][:, i].float().to(C.DEV)
    gam = nb.block[2].weight.detach(); kap = nb.block[0].weight.detach().sum((1, 2, 3))
    if w.shape[0] != gam.shape[0]: return []           # across the stage-3 -> stage-4 downsample: skip
    e = nb.block[3].weight.detach() @ (gam * kap * w); top = torch.argsort(-e.abs())[:5]
    return [(int(k), float(e[k])) for k in top]
def edges_in(g, i):
    if g == 0 or g == 27: return []
    pb = BL[g - 1]; Wb = C.write_mat(pb)[1].float().to(C.DEV)            # (d, h_prev)
    blk = BL[g]; gam = blk.block[2].weight.detach(); kap = blk.block[0].weight.detach().sum((1, 2, 3))
    e = (blk.block[3].weight.detach()[i] * gam * kap) @ Wb; top = torch.argsort(-e.abs())[:5]
    return [(int(k), float(e[k])) for k in top]
track = {}                                   # (g, base index) -> role
for n in NE:
    track[(n["g"], n["base_index"])] = n["name"]
    n["feeders"] = edges_in(n["g"], n["base_index"]); n["readers"] = edges_out(n["g"], n["base_index"])
    for k, _ in n["feeders"]: track.setdefault((n["g"] - 1, k), f"{n['name']}.feeder")
    for k, _ in n["readers"]: track.setdefault((n["g"] + 1, k), f"{n['name']}.reader")
keys = sorted(track)
say(f"{len(keys)} neurons tracked")
def run(m, which, ix, mode):
    """mode 'base': Base's own neurons. mode 'net': B or T; survivors at their positions, others virtual (Base read row)."""
    store = {}
    hs = []
    for g in sorted({g for g, _ in keys}):
        idx = [b for gg, b in keys if gg == g]
        if mode == "base":
            hs.append(C.blocks(m)[g].block[3].register_forward_hook(lambda mod, i, o, g=g, idx=idx: store.__setitem__(g, o[..., idx])))
        else:
            W = L1(base, g).weight[idx].detach(); bb = L1(base, g).bias[idx].detach()
            def f(mod, i, o, g=g, idx=idx, W=W, bb=bb):
                v = i[0] @ W.T + bb                             # virtual value from Base's read row
                for c, b in enumerate(idx):
                    if b in pos[g]: v[..., c] = o[..., pos[g][b]]
                store[g] = v
            hs.append(C.blocks(m)[g].block[3].register_forward_hook(f))
    ph = C.net.Photos("read"); head = torch.nn.Sequential(*[base.features[i] for i in range(5)])
    N, K = len(ix), len(keys)
    mx, mn = np.zeros((N, K), np.float16), np.zeros((N, K), np.float16); amx, amn = np.zeros((N, K), np.int16), np.zeros((N, K), np.int16)
    col = {g: [c for c, (gg, _) in enumerate(keys) if gg == g] for g in {g for g, _ in keys}}
    with torch.no_grad():
        for s in range(0, N, 50):
            C.tail(m, head(ph.tensor(ix[s:s + 50]).to(C.DEV)))
            for g, cs in col.items():
                v = store[g].flatten(1, 2)                      # (b, positions, k)
                a, ai = v.max(1); b_, bi = v.min(1)
                mx[s:s + 50, cs] = a.cpu().numpy(); mn[s:s + 50, cs] = b_.cpu().numpy()
                amx[s:s + 50, cs] = ai.cpu().numpy(); amn[s:s + 50, cs] = bi.cpu().numpy()
            if s % 5000 == 0: say(which, s, N)
            if "--smoke" in sys.argv and s >= 100: break
    for h in hs: h.remove()
    np.savez(os.path.join(OUT, f"pilot_{which}.npz"), ix=ix, mx=mx, mn=mn, amx=amx, amn=amn)
sets = C.photo_sets(LAB); trained = set(sets["train"].tolist()) | set(sets["cal"].tolist())
safe = np.array([i for i in range(len(LAB)) if i not in trained])        # held + test + unused odd photos
run(base, "base", np.arange(len(LAB)), "base")
run(mB, "B", safe, "net"); run(mT, "T", safe, "net")
json.dump(dict(neurons=NE, keys=[list(k) + [track[k]] for k in keys]), open(os.path.join(OUT, "pilot_meta.json"), "w"), indent=1)
say("done")
