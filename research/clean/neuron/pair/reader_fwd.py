"""reader, step 1: 30 counterpart pairs (s, p), block 0, and what Base has on in their rows.

Pairs: Small s whose firing match (tools_peakR_b0 argmax) and writing match (writesR_b0 argmax) are the same Base p.
30 pairs spread evenly over the firing correlation R(s, p), one per distinct p.
Photos: the 4,000 topsets photos (fit then test), rows = photo x position (7 x 7).
Pass 1: pre-activations (block[3] output) of the 30 s and 30 p at every row.
Row sets per pair (thresholds over all 196,000 rows):
  s_top   s in its top 1%                 p_top   p in its top 1%
  s_only  s in its top 1% and p < 0       p_only  p in its top 1% and s < 0
  s_bot   s in its bottom 1% (negative pole)   p_bot   p in its bottom 1%
Pass 2: mean post-GELU of all 4,096 Base block-0 neurons and all 3,072 Small block-0 neurons over each set,
plus the mean and sd over all rows (to z-score).
Output: results/pair/reader/fwd.npz
"""
import argparse
import os
import time

import numpy as np
import torch

from cache import DEV, OUT, load, net

torch.set_grad_enabled(False)
RD = os.path.join(OUT, "reader")
SETS = ["s_top", "p_top", "s_only", "p_only", "s_bot", "p_bot"]


def pick_pairs(n=30):
    Rf = np.load(os.path.join(OUT, "tools_peakR_b0.npy")).astype(np.float32)
    Rw = np.load(os.path.join(OUT, "writesR_b0.npy")).astype(np.float32)
    af, aw = Rf.argmax(1), Rw.argmax(1)
    s_ok = np.where(af == aw)[0]
    r = Rf[s_ok, af[s_ok]]
    o = s_ok[np.argsort(r)]
    rng = np.random.default_rng(0)
    out, used = [], set()
    for q in np.linspace(0, len(o) - 1, 4 * n).astype(int):
        s = int(o[q])
        p = int(af[s])
        if p in used:
            continue
        out.append((s, p, float(Rf[s, p]), float(Rw[s, p])))
        used.add(p)
    # keep n evenly spaced
    keep = np.linspace(0, len(out) - 1, n).astype(int)
    return [out[k] for k in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(RD, exist_ok=True)
    pairs = pick_pairs()
    S_ix = torch.tensor([q[0] for q in pairs], device=DEV)
    P_ix = torch.tensor([q[1] for q in pairs], device=DEV)
    d = np.load(os.path.join(OUT, "topsets.npz"))
    photos = np.concatenate([d["fit_photos"], d["test_photos"]])
    if a.n:
        photos = photos[:a.n]
    ph = net.Photos("read")
    ms, mb = load("small"), load("base")
    H = {}
    ms.features[7][0].block[3].register_forward_hook(lambda m, i, o: H.__setitem__("s", o))
    mb.features[7][0].block[3].register_forward_hook(lambda m, i, o: H.__setitem__("b", o))
    B = 50
    t0 = time.time()
    # pass 1
    SV, PV = [], []
    for k in range(0, len(photos), B):
        x = ph.tensor(photos[k:k + B]).to(DEV)
        ms(x), mb(x)
        SV.append(H["s"].reshape(-1, 3072)[:, S_ix].float().cpu())
        PV.append(H["b"].reshape(-1, 4096)[:, P_ix].float().cpu())
        if (k // B) % 20 == 0:
            print(f"pass1 {k + B}/{len(photos)} {time.time() - t0:.0f}s", flush=True)
    SV, PV = torch.cat(SV).numpy(), torch.cat(PV).numpy()
    N = SV.shape[0]
    K = max(int(N * 0.01), 10)
    masks = {}
    for i in range(len(pairs)):
        s, p = SV[:, i], PV[:, i]
        st = np.zeros(N, bool); st[np.argsort(-s)[:K]] = True
        pt = np.zeros(N, bool); pt[np.argsort(-p)[:K]] = True
        sb = np.zeros(N, bool); sb[np.argsort(s)[:K]] = True
        pb = np.zeros(N, bool); pb[np.argsort(p)[:K]] = True
        masks[i] = dict(s_top=st, p_top=pt, s_only=st & (p < 0), p_only=pt & (s < 0), s_bot=sb, p_bot=pb)
    M = torch.tensor(np.stack([np.stack([masks[i][n] for n in SETS]) for i in range(len(pairs))]),
                     dtype=torch.float32)  # (30, 6, N)
    M = M.reshape(-1, N)
    # pass 2
    accB = torch.zeros(M.shape[0], 4096, dtype=torch.float64)
    accS = torch.zeros(M.shape[0], 3072, dtype=torch.float64)
    sb_, sbb, ss_, sss = [torch.zeros(n, dtype=torch.float64) for n in (4096, 4096, 3072, 3072)]
    gelu = torch.nn.GELU()
    for k in range(0, len(photos), B):
        x = ph.tensor(photos[k:k + B]).to(DEV)
        ms(x), mb(x)
        gb = gelu(H["b"].reshape(-1, 4096).float())
        gs = gelu(H["s"].reshape(-1, 3072).float())
        r0 = k * 49
        m = M[:, r0:r0 + gb.shape[0]].to(DEV)
        accB += (m @ gb).cpu().double()
        accS += (m @ gs).cpu().double()
        sb_ += gb.sum(0).cpu().double(); sbb += (gb * gb).sum(0).cpu().double()
        ss_ += gs.sum(0).cpu().double(); sss += (gs * gs).sum(0).cpu().double()
        if (k // B) % 20 == 0:
            print(f"pass2 {k + B}/{len(photos)} {time.time() - t0:.0f}s", flush=True)
    cnt = M.sum(1, keepdim=True).double().clamp_min(1)
    mB, mS = sb_ / N, ss_ / N
    sdB = (sbb / N - mB ** 2).clamp_min(1e-12).sqrt()
    sdS = (sss / N - mS ** 2).clamp_min(1e-12).sqrt()
    zB = ((accB / cnt - mB) / sdB).reshape(len(pairs), len(SETS), 4096).numpy().astype(np.float32)
    zS = ((accS / cnt - mS) / sdS).reshape(len(pairs), len(SETS), 3072).numpy().astype(np.float32)
    np.savez(os.path.join(RD, "fwd_smoke.npz" if a.n else "fwd.npz"),
             pairs=np.array(pairs), photos=photos, SV=SV.astype(np.float16), PV=PV.astype(np.float16),
             counts=cnt.reshape(len(pairs), len(SETS)).numpy(), zB=zB, zS=zS, sets=np.array(SETS))
    print("done", time.time() - t0, "counts s_only:", cnt.reshape(len(pairs), len(SETS))[:, 2].numpy().astype(int))


if __name__ == "__main__":
    main()
