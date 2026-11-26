# scale: neurons=A:block0-all, A:block1-weights, A:head-weights; net=A, Z-ladder, C-ladder, W-ladder, WC-ladder; photos=held, test; where=cluster
# rung: Q=plan "Read what training adds on top of the calculation", steps 1b+4+5: which part of training's write change carries the KL gain, quadratic forecast, harm bins; N=4 cuts x 9 builds, forward only; EST=15 min GPU; PERMIT=Shonu approved plan 2026-11-25
"""Cluster job, forward passes only (no training). Runs in n6/clean/neuron/coarse next to r18.py and oc_retrain.py.

Step 1b (causal split of WC - C). Step 1 (oc_vs_writes.py) split training's change of the kept write columns, in the
effective write space V = bn2 scale x conv2, into Dspan (A's deleted writes), Kspan (A's kept writes, outside Dspan) and
rest (outside every A write). Here each part is added to C alone, and the bn2 shift change alone:
  C+D, C+K, C+rest, C+DK, C+shift, C+all (= all write parts, C's shift), and C+all+shift (must equal WC exactly).
  Leave-one-out with WC's shift: C+DK+shift (WC without rest), C+rest+shift (WC without the D and K parts).
Scored by KL(A || build) on all 17,000 test photos.
Follow-up (argv spatial, added after the first run): WC - C split into its 3 x 3-uniform part (same 512-channel change at all 9 positions) and its spatial part (sums to 0 over the 9); also the rest part split the same way. Bar written before this run: "the gain is mostly spatial" holds if g(C+spat+shift) >= .5 and > g(C+uni+shift).
Bars, written 2026-11-25 before the run. Gain share g(h) = (KL_C - KL_h) / (KL_C - KL_WC):
  rest = optimizer drift      if g(C+DK+shift) >= .90 at a cut
  rest carries the gain       if g(C+rest+shift) >= .50 at a cut
  neither                     the parts work only together (report as such)

Step 5. For C, W, WC and the hybrids, per test photo:
  e2   = |e|^2, e = stream after block 0 (before the ReLU, r + skip) of the build minus A's
  q    = 1/2 (J e)^T F (J e): the quadratic KL forecast (J = A's tail from that stream to the logits, F = A's softmax
         Fisher diag(p) - p p^T)
  kl   = KL(A || build)
  eG   = sum over cells of e^T G e / trace(G), G = readers' Fisher at the stream, channel-pooled (512 x 512), from held
         photos with one sampled label each (true Fisher, not empirical)
  top  = share of |e|^2 along G's top 20 eigenvectors
Step 4 data: A's summed firing of each cut's D per test photo, and the class label.
Saved: results/coarse/r18/oc_vs/ (why.json, per-photo arrays n<cut>.npz, G.npy).
Usage: python -u oc_vs_why.py [smoke]
"""
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import oc_retrain as OR
import r18

SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
SPATIAL = "spatial" in sys.argv     # follow-up run: only the uniform / spatial split of WC - C (added 2026-11-25)
DEV = OR.DEV
CUTS = (11, 384) if SMOKE else (11, 64, 192, 384)
LAD = os.path.join(r18.OUT, "oc_ladder")
OUT = os.path.join(r18.OUT, "oc_vs" + ("_smoke" if SMOKE else "") + ("_spatial" if SPATIAL else ""))
TOPK = 20
T0 = time.time()


def log(*a):
    print(*a, f"[{time.time() - T0:.0f} s]", flush=True)


def ranking(a):
    b0 = a.layer4[0]
    s1, _ = OR.bn_parts(b0.bn1)
    s2, _ = OR.bn_parts(b0.bn2)
    R = b0.conv1.weight * s1.view(-1, 1, 1, 1)
    V = b0.conv2.weight * s2.view(-1, 1, 1, 1)
    score = (R[:, OR.P].sum((1, 2)) * torch.sqrt((V ** 2).sum((0, 2, 3)))).detach().cpu().numpy()
    return np.argsort(-score)


def load_build(A, n, b):
    m = copy.deepcopy(A)
    m.load_state_dict(torch.load(os.path.join(LAD, f"n{n}", f"{b}.pt"), map_location=DEV, weights_only=True))
    return m.eval()


def eff(m):
    s2, t2 = OR.bn_parts(m.layer4[0].bn2)
    return (m.layer4[0].conv2.weight * s2.view(-1, 1, 1, 1)).detach().double(), t2.detach().double()


def with_writes(C, Vt, dshift):
    """C with effective writes Vt (C's bn2 scale kept, so conv2 = Vt / s2_C) and bn2 shift + dshift."""
    m = copy.deepcopy(C)
    s2, _ = OR.bn_parts(C.layer4[0].bn2)
    with torch.no_grad():
        m.layer4[0].conv2.weight.copy_((Vt / s2.double().view(-1, 1, 1, 1)).float())
        m.layer4[0].bn2.bias.add_(dshift.float())
    return m.eval()


def basis(M):
    q, r = torch.linalg.qr(M.T)
    d = r.diagonal().abs()
    return q[:, d > 1e-10 * d.max()]


def s4b0(m, x3):
    b0 = m.layer4[0]
    return b0.bn2(b0.conv2(torch.relu(b0.bn1(b0.conv1(x3))))) + b0.downsample(x3)


def tail(m, s):
    return m.fc(torch.flatten(m.avgpool(m.layer4[1](torch.relu(s))), 1))


@torch.no_grad()
def stage3(A, U, bs=250):
    return torch.cat([A.layer3(A.layer2(A.layer1(r18.net.stem(A, OR.norm(U[a:a + bs]))))).half()
                      for a in range(0, len(U), bs)])


def fisher(A, X3h, bs=100):
    """Channel-pooled readers' Fisher at A's stream after block 0, one sampled label per photo."""
    g = torch.Generator(device=DEV).manual_seed(0)
    G = torch.zeros(512, 512, dtype=torch.float64, device=DEV)
    for a in range(0, len(X3h), bs):
        with torch.no_grad():
            s = s4b0(A, X3h[a:a + bs].float())
        s = s.detach().requires_grad_(True)
        lp = F.log_softmax(tail(A, s).float(), 1)
        y = torch.multinomial(lp.exp(), 1, generator=g).squeeze(1)
        gr, = torch.autograd.grad(lp.gather(1, y[:, None]).sum(), s)
        v = gr.permute(0, 2, 3, 1).reshape(-1, 512).double()
        G += v.T @ v
    return G / len(X3h)


def measure(A, m, X3h, ZA, U_top, G, bs=100):
    """Per test photo: e2, q, kl, eG, top share."""
    out = {k: [] for k in ("e2", "q", "kl", "eG", "top")}
    trG = float(G.trace())
    for a in range(0, len(X3h), bs):
        x = X3h[a:a + bs].float()
        with torch.no_grad():
            sA, sM = s4b0(A, x), s4b0(m, x)
            e = sM - sA
            zm = tail(m, sM)
        _, Je = torch.func.jvp(lambda s: tail(A, s), (sA,), (e,))
        p = F.softmax(ZA[a:a + bs].float(), 1)
        Je = Je.float()
        q = 0.5 * ((Je ** 2 * p).sum(1) - (Je * p).sum(1) ** 2)
        ev = e.permute(0, 2, 3, 1).reshape(e.shape[0], -1, 512).double()
        e2 = (ev ** 2).sum((1, 2))
        eG = torch.einsum("bci,ij,bcj->b", ev, G, ev) / trG
        top = ((ev @ U_top) ** 2).sum((1, 2)) / e2.clamp_min(1e-30)
        out["e2"].append(e2.float()); out["q"].append(q); out["eG"].append(eG.float()); out["top"].append(top.float())
        out["kl"].append(r18.kl_rows(ZA[a:a + bs], zm))
    return {k: torch.cat(v).cpu().numpy() for k, v in out.items()}


def main():
    os.makedirs(OUT, exist_ok=True)
    A = r18.load_a().eval()
    for p in A.parameters():
        p.requires_grad_(False)
    order = ranking(A)
    assert sorted(order[:11].tolist()) == sorted([305, 97, 23, 163, 385, 297, 464, 237, 108, 510, 275])
    ph = r18.net.Photos("read")
    sets = r18.photo_sets(ph.lab)
    if SMOKE:
        sets = {k: v[:: {"train": 31, "held": 15, "test": 66}[k]] for k, v in sets.items()}
    U = {k: OR.load_photos(ph, sets[k]) for k in ("held", "test")}
    labels = np.asarray(ph.lab)[sets["test"]]
    ZA = OR.logits(A, U["test"])
    X3 = {k: stage3(A, U[k]) for k in ("held", "test")}
    del U
    log("photos:", {k: len(v) for k, v in X3.items()})
    # check: the split forward equals A's own logits
    with torch.no_grad():
        zz = torch.cat([tail(A, s4b0(A, X3["test"][a:a + 250].float())) for a in range(0, len(X3["test"]), 250)])
    chk = float(r18.kl_rows(ZA, zz).max())
    log(f"check: KL(A full || A via cached stage 3) max {chk:.1e}")
    ZA = zz                      # reference = the cached path, so KL(A||A) = 0 exactly and q expands at the same point

    G = fisher(A, X3["held"])
    ev, evec = torch.linalg.eigh(G)
    U_top = evec[:, -TOPK:]
    np.save(os.path.join(OUT, "G.npy"), G.cpu().numpy())
    log(f"Fisher G: trace {float(G.trace()):.3e}; top-{TOPK} share of trace {float(ev[-TOPK:].sum() / ev.sum()):.3f}")

    VA, tA = eff(A)
    res = dict(check_split_forward=chk, G_top_share=float(ev[-TOPK:].sum() / ev.sum()), cuts={})
    mA = measure(A, A, X3["test"], ZA, U_top, G)
    log(f"check A: max e2 {mA['e2'].max():.1e}, max q {mA['q'].max():.1e}, max kl {mA['kl'].max():.1e}")
    for n in CUTS:
        D = sorted(order[:n].tolist())
        K = np.setdiff1d(np.arange(512), D).tolist()
        Z = copy.deepcopy(A)
        OR.apply_masks(Z, OR.masks(Z, D))
        builds = {"Z": Z.eval()}
        for b in ("C", "W", "WC"):
            builds[b] = load_build(A, n, b)
        C, WC = builds["C"], builds["WC"]
        VC, tC = eff(C)
        VW, tW = eff(WC)
        # split of WC - C per kept write column (rows = input neuron j, 4608 numbers)
        rows = lambda V: V.permute(1, 0, 2, 3).reshape(512, -1)
        Dl = rows(VW - VC)[K]
        QD = basis(rows(VA)[D])
        RK = rows(VA)[K]
        QK = basis(RK - (RK @ QD) @ QD.T)
        pD = (Dl @ QD) @ QD.T
        pK = ((Dl - pD) @ QK) @ QK.T
        parts = {"D": pD, "K": pK, "rest": Dl - pD - pK}
        zero = torch.zeros_like(tC)

        def full(add):
            R = rows(VC).clone()
            R[K] += add
            return R.reshape(512, 512, 3, 3).permute(1, 0, 2, 3)

        hyb = {"C+D": (parts["D"], zero), "C+K": (parts["K"], zero), "C+rest": (parts["rest"], zero),
               "C+DK": (parts["D"] + parts["K"], zero), "C+shift": (0 * Dl, tW - tC),
               "C+all": (Dl, zero), "C+all+shift": (Dl, tW - tC),
               "C+DK+shift": (parts["D"] + parts["K"], tW - tC), "C+rest+shift": (parts["rest"], tW - tC)}
        uni = lambda P: P.reshape(P.shape[0], 512, 9).mean(2, keepdim=True).expand(-1, -1, 9).reshape(P.shape[0], -1)
        if SPATIAL:
            hyb = {"C+all+shift": (Dl, tW - tC), "C+uni+shift": (uni(Dl), tW - tC),
                   "C+spat+shift": (Dl - uni(Dl), tW - tC), "C+restuni+shift": (uni(parts["rest"]), tW - tC),
                   "C+restspat+shift": (parts["rest"] - uni(parts["rest"]), tW - tC)}
            for b in ("W",):
                builds.pop(b)
        for h, (add, dsh) in hyb.items():
            builds[h] = with_writes(C, full(add), dsh)
        b0A = A.layer4[0]
        with torch.no_grad():
            dfire = torch.cat([torch.relu(b0A.bn1(b0A.conv1(X3["test"][a:a + 250].float())))[:, D].sum((1, 2, 3))
                               for a in range(0, len(X3["test"]), 250)]).cpu().numpy()
        r = dict(D=D, parts_share={k: float((v ** 2).sum() / (Dl ** 2).sum()) for k, v in parts.items()}, builds={})
        arrays = dict(labels=labels, dfire=dfire)
        klZ = None
        for b, m in builds.items():
            mm = measure(A, m, X3["test"], ZA, U_top, G)
            if b == "Z":
                klZ = mm["kl"].mean()
            row = {k: float(v.mean()) for k, v in mm.items()}
            row["rec"] = float(1 - mm["kl"].mean() / klZ)
            row["q_over_kl"] = float(mm["q"].mean() / mm["kl"].mean())
            r["builds"][b] = row
            for k, v in mm.items():
                arrays[f"{b}__{k}"] = v.astype(np.float32)
            log(f"n{n} {b:12s} rec {row['rec']:.3f}  kl {row['kl']:.4f}  q/kl {row['q_over_kl']:.2f}  "
                f"e2 {row['e2']:.2f}  eG {row['eG']:.3e}  top{TOPK} {row['top']:.3f}")
        # checks against the ladder and exactness
        r["check_WC_equal"] = float(abs(r["builds"]["C+all+shift"]["kl"] - r["builds"]["WC"]["kl"]))
        kc, kw = r["builds"]["C"]["kl"], r["builds"]["WC"]["kl"]
        r["gain_share"] = {h: float((kc - r["builds"][h]["kl"]) / (kc - kw)) for h in builds if h.startswith("C+")}
        log(f"n{n} gain share of WC - C: " + ", ".join(f"{h} {v:.2f}" for h, v in r["gain_share"].items()))
        for b in ("Z", "C", "W", "WC"):
            f = os.path.join(LAD, f"n{n}", f"kl_test_{b}.npy")
            if os.path.exists(f) and not SMOKE and b in r["builds"]:
                r[f"check_{b}_vs_ladder"] = float(abs(np.load(f).mean() - r["builds"][b]["kl"]))
        log(f"n{n} checks: " + ", ".join(f"{k} {v:.1e}" for k, v in r.items() if k.startswith("check")))
        res["cuts"][str(n)] = r
        np.savez(os.path.join(OUT, f"n{n}.npz"), **arrays)
        json.dump(res, open(os.path.join(OUT, "why.json"), "w"), indent=1)
        del builds
    log("saved", OUT)


if __name__ == "__main__":
    main()
