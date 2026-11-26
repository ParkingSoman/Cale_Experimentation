# scale: neurons=s3:69, A:children, A:all-units; net=A, R, R4; photos=held; where=cluster
# rung: Q=who did R's work: path scores by unit, effective writes of stage-3 channels, swap-in / revert; N=1 net, 2 data orders; EST=15 min GPU; PERMIT=Shonu approved plan "Delete set D: train R, read it neuron by neuron" 2026-11-24
"""Rung 2 of the plan "Delete set D" (2026-11-24). Reads R (from oc_retrain.py) unit by unit. Held photos only.

1. Path score. theta(t) = Z + t (R - Z). Score of a weight = (R - Z) x mean over t of dKL/dtheta, KL = KL(A || net).
   Midpoint rule, 16 steps. The scores sum to KL(R) - KL(Z). Recovery share of a group = - its sum / (KL(Z) - KL(R)).
   Each weight is an edge sender -> receiver. Units:
     s0:c        stem output channel             s{S}b{b}n:j  inner neuron of stage S block b
     s{S}b{b}:c  stream point after stage S block b (s3b1:c = "s3:c")      out:k  head row
   conv1 of (S, b): sender = its input stream point, receiver = inner. bn1 -> inner.
   conv2: sender = inner, receiver = stream point. bn2, downsample bn -> stream point. downsample conv: sender = input
   stream point, receiver = stream point. stem conv: receiver s0. fc: sender s4b1, receiver out.
   Groups (set before the run): P = layer4.0.conv1.weight + layer4.0.downsample.0.weight (edges out of stage-3 channels
   into stage-4 block 0). S = layer4.0 bn1, conv2, bn2, downsample bn. W = layer4.1, fc. U = everything else.
2. Effective write of a stage-3 channel c in net m: d(stage-4 block-0 stream z, before the ReLU) / d(channel c raised
   by the same small amount at every cell), summed over the 7 x 7 cells -> 512 numbers, per photo (each net's own
   stage-3 output). Exact JVP. Saved: mean and sd over photos for all 256 channels in Z, R_s0, R_s1, and per photo for
   s3:69 and the top 30 senders. Also the single-parent calculated write (oc_morph, photo independent), reference only.
3. Swap-in (Z + one unit's R weights) and revert (R with one unit back to Z): top 20 senders (P) and top 20 receivers.
4. Reads and writes for the ledger: block-0 kept neurons' read rows (bn1 folded) and write columns summed over 3 x 3
   (bn2 folded), in Z and R. The children's summed write columns (for "share inside the children's paths").
Usage: python -u oc_respond.py [smoke]
"""
import copy
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp

import r18

SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
DEV = r18.DEV
P = 69
LIGHT = "light" in sys.argv                # 8 path steps, 1,000 held photos (faster CPU run); path-sum check still applies
NT = 4 if SMOKE else (8 if LIGHT else 16)
NPH = 64 if SMOKE else (1000 if LIGHT else 2000)
NJ = 32 if SMOKE else 256
TOPX = 20
SRC = os.path.join(r18.OUT, "oc_retrain" + ("_smoke" if SMOKE else ""))
OUT = SRC
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
T0 = time.time()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from oc_retrain import bn_parts, children, masks, apply_masks, load_photos, norm  # noqa: E402


def log(*a):
    print(*a, f"[{time.time() - T0:.0f} s]", flush=True)


def stream_in(S, b):
    if b == 1:
        return f"s{S}b0"
    return "s0" if S == 1 else f"s{S - 1}b1"


def roles(name):
    """(sender layer name or None, sender dim, receiver layer name, receiver dim, group)."""
    if name.startswith("fc."):
        return ("s4b1", 1, "out", 0, "W") if name.endswith("weight") else (None, None, "out", 0, "W")
    if name.startswith("conv1.") or name.startswith("bn1."):
        return (None, None, "s0", 0, "U")
    m = re.match(r"layer(\d)\.(\d)\.(.+)", name)
    S, b, rest = int(m.group(1)), int(m.group(2)), m.group(3)
    g = "U" if S < 4 else ("W" if b == 1 else None)
    inner, point, src = f"s{S}b{b}n", f"s{S}b{b}", stream_in(S, b)
    if rest == "conv1.weight":
        return (src, 1, inner, 0, g or "P")
    if rest.startswith("bn1."):
        return (None, None, inner, 0, g or "S")
    if rest == "conv2.weight":
        return (inner, 1, point, 0, g or "S")
    if rest.startswith("bn2."):
        return (None, None, point, 0, g or "S")
    if rest == "downsample.0.weight":
        return (src, 1, point, 0, g or "P")
    if rest.startswith("downsample.1."):
        return (None, None, point, 0, g or "S")
    raise ValueError(name)


def kl_mean(m, U, ZA, params=None, bs=250):
    tot = 0.0
    for a in range(0, len(U), bs):
        x = norm(U[a:a + bs])
        z = m(x) if params is None else functional_call(m, params, (x,))
        tot = tot + r18.kl_rows(ZA[a:a + bs], z.float()).sum()
    return tot / len(U)


def path_scores(m, pz, pr, U, ZA):
    names = list(pz)
    d = {n: pr[n] - pz[n] for n in names}
    g = {n: torch.zeros_like(pz[n]) for n in names}
    for k in range(NT):
        t = (k + 0.5) / NT
        th = {n: (pz[n] + t * d[n]).requires_grad_(True) for n in names}
        for a in range(0, len(U), 125):
            z = functional_call(m, th, (norm(U[a:a + 125]),))
            loss = r18.kl_rows(ZA[a:a + 125], z.float()).sum() / len(U)
            gr = torch.autograd.grad(loss, [th[n] for n in names])
            for n, x in zip(names, gr):
                g[n] += x / NT
    return {n: (d[n] * g[n]).detach() for n in names}


def group_units(score):
    send, recv, grp = {}, {}, {"P": 0.0, "S": 0.0, "W": 0.0, "U": 0.0}
    for n, s in score.items():
        sl, sd, rl, rd, gname = roles(n)
        grp[gname] += float(s.sum())
        if s.dim() == 0:
            continue
        rv = s.sum(dim=[i for i in range(s.dim()) if i != rd]) if s.dim() > 1 else s
        recv[rl] = recv.get(rl, 0) + rv.cpu().numpy().astype(np.float64)
        if sl is not None:
            sv = s.sum(dim=[i for i in range(s.dim()) if i != sd])
            key = (sl, gname)
            send[key] = send.get(key, 0) + sv.cpu().numpy().astype(np.float64)
    return send, recv, grp


def unit_slices(model, kind, layer, idx):
    """Parameter slices that belong to one unit: kind 'send' (its outgoing edges) or 'recv' (its read)."""
    out = []
    for n, p in model.named_parameters():
        sl, sd, rl, rd, _ = roles(n)
        if kind == "send" and sl == layer and sd is not None:
            out.append((n, sd))
        if kind == "recv" and rl == layer:
            out.append((n, rd))
    return [(n, d, idx) for n, d in out]


def swapped(base, other, slices):
    m = copy.deepcopy(base)
    po = dict(other.named_parameters())
    with torch.no_grad():
        for n, d, i in slices:
            p = dict(m.named_parameters())[n]
            p.index_copy_(d, torch.tensor([i], device=DEV), po[n].index_select(d, torch.tensor([i], device=DEV)))
    return m


def block0_z(b0, h):
    inner = torch.relu(b0.bn1(b0.conv1(h)))
    return b0.bn2(b0.conv2(inner)) + b0.downsample(h)


@torch.no_grad()
def stage3(m, x):
    return m.layer3(m.layer2(m.layer1(r18.net.stem(m, x))))


def eff_writes(m, U, chans, bs=8):
    """Per photo: d z (summed over cells) / d (uniform rise of channel c). Returns photos x len(chans) x 512."""
    b0 = m.layer4[0]
    out = []
    for a in range(0, len(U), bs):
        h = stage3(m, norm(U[a:a + bs]))
        n = h.shape[0]
        hs = h.unsqueeze(1).expand(n, len(chans), *h.shape[1:]).reshape(-1, *h.shape[1:])
        tg = torch.zeros_like(hs).view(n, len(chans), *h.shape[1:])
        for j, c in enumerate(chans):
            tg[:, j, c] = 1.0
        tg = tg.view_as(hs)
        _, dz = jvp(lambda u: block0_z(b0, u), (hs,), (tg,))
        out.append(dz.sum((2, 3)).view(n, len(chans), 512).float().cpu())
    return torch.cat(out).numpy()


def calc_write(a, D):
    """Single-parent calculated write of s3:69 (oc_morph), summed over cells, for a uniform unit rise: 512 numbers."""
    b0 = a.layer4[0]
    s1, _ = bn_parts(b0.bn1)
    s2, _ = bn_parts(b0.bn2)
    K = torch.as_tensor(D, device=DEV)
    k = (b0.conv1.weight * s1.view(-1, 1, 1, 1))[K][:, P:P + 1]
    v = (b0.conv2.weight * s2.view(-1, 1, 1, 1))[:, K]
    one = torch.ones(1, 1, 14, 14, device=DEV)
    return F.conv2d(F.conv2d(one, k, stride=2, padding=1), v, padding=1).sum((2, 3))[0].detach().cpu().numpy()


def main():
    A = r18.load_a()
    D = children(A)
    ph = r18.net.Photos("read")
    held = r18.photo_sets(ph.lab)["held"][:NPH]
    U = load_photos(ph, held)
    with torch.no_grad():
        ZA = torch.cat([A(norm(U[a:a + 250])).float() for a in range(0, len(U), 250)])
    Z = copy.deepcopy(A)
    apply_masks(Z, masks(Z, D))
    nets = {"Z": Z}
    for tag in ("R_s0", "R_s1", "R4_s0"):
        m = copy.deepcopy(A)
        m.load_state_dict(torch.load(os.path.join(SRC, f"{tag}.pt"), weights_only=True))
        nets[tag] = m.eval()
    with torch.no_grad():
        kl = {t: float(kl_mean(m, U, ZA)) for t, m in nets.items()}
    log("held KL:", {t: round(v, 5) for t, v in kl.items()})
    pz = {n: p.detach() for n, p in Z.named_parameters()}

    res = dict(D=D, kl=kl, runs={})
    save = {}
    for tag in ("R_s0", "R_s1", "R4_s0"):
        pr = {n: p.detach() for n, p in nets[tag].named_parameters()}
        sc = path_scores(Z, pz, pr, U, ZA)
        tot = sum(float(s.sum()) for s in sc.values())
        want = kl[tag] - kl["Z"]
        send, recv, grp = group_units(sc)
        gap = kl["Z"] - kl[tag]
        share = {g: -v / gap for g, v in grp.items()}
        log(f"{tag}: path sum {tot:.6f} vs KL(R)-KL(Z) {want:.6f} (error {abs(tot - want) / abs(want):.3%}); "
            f"group shares {({g: round(v, 3) for g, v in share.items()})}")
        res["runs"][tag] = dict(path_sum=tot, want=want, share=share,
                                per_param={n: -float(v.sum()) / gap for n, v in sc.items()})
        for (layer, g), v in send.items():
            save[f"{tag}__send__{layer}__{g}"] = -v / gap
        for layer, v in recv.items():
            save[f"{tag}__recv__{layer}"] = -v / gap

    # top senders (P, stage-3 channels) and top receivers (all units) in R_s0
    sP = save["R_s0__send__s3b1__P"]
    top_send = np.argsort(-np.abs(sP))[:TOPX].tolist()
    allr = [(layer, j, v) for k, arr in save.items() if k.startswith("R_s0__recv__")
            for layer in [k.split("__")[2]] for j, v in enumerate(arr)]
    allr.sort(key=lambda r: -abs(r[2]))
    top_recv = [(layer, j) for layer, j, _ in allr[:TOPX]]
    swaps = []
    R0 = nets["R_s0"]
    for kind, items in (("send", [("s3b1", c) for c in top_send]), ("recv", top_recv)):
        for layer, j in items:
            sl = unit_slices(Z, kind, layer, j)
            if kind == "send":
                sl = [(n, d, i) for n, d, i in sl if roles(n)[4] == "P"]
            with torch.no_grad():
                kin = float(kl_mean(swapped(Z, R0, sl), U, ZA))
                kout = float(kl_mean(swapped(R0, Z, sl), U, ZA))
            gap = kl["Z"] - kl["R_s0"]
            path = float(save[f"R_s0__send__s3b1__P"][j]) if kind == "send" else float(save[f"R_s0__recv__{layer}"][j])
            swaps.append(dict(kind=kind, unit=f"{layer}:{j}", path=path,
                              swap_in=(kl["Z"] - kin) / gap, revert=(kout - kl["R_s0"]) / gap))
    agree = np.mean([np.sign(s["path"]) == np.sign(s["swap_in"]) for s in swaps])
    res["swaps"] = swaps
    res["swap_sign_agree"] = float(agree)
    log(f"swap-in sign agrees with path score on {agree:.2f} of {len(swaps)} units")

    # effective writes of stage-3 channels
    chans = list(range(NJ))
    Ue = U[:256 if not SMOKE else 16]
    ew = {}
    for tag in ("Z", "R_s0", "R_s1"):
        w = np.concatenate([eff_writes(nets[tag], Ue, chans[a:a + 32]) for a in range(0, NJ, 32)], axis=1)
        ew[tag] = w
        save[f"effw_mean__{tag}"] = w.mean(0)
        save[f"effw_sd__{tag}"] = w.std(0)
    keep = sorted(set(top_send + [P] + np.argsort(-np.abs(sP))[:30].tolist()))
    keep = [c for c in keep if c < NJ]
    save["effw_keep"] = np.array(keep)
    for tag in ("Z", "R_s0", "R_s1"):
        save[f"effw_photo__{tag}"] = ew[tag][:, keep].astype(np.float16)
    save["calc_write_69"] = calc_write(A, D)

    # reads and writes of block-0 neurons, Z and R
    for tag in ("Z", "R_s0", "R_s1"):
        b0 = nets[tag].layer4[0]
        s1, _ = bn_parts(b0.bn1)
        s2, _ = bn_parts(b0.bn2)
        save[f"read__{tag}"] = (b0.conv1.weight * s1.view(-1, 1, 1, 1)).detach().cpu().numpy().astype(np.float16)
        save[f"write__{tag}"] = (b0.conv2.weight * s2.view(-1, 1, 1, 1)).sum((2, 3)).detach().cpu().numpy()
    b0 = A.layer4[0]
    s2, _ = bn_parts(b0.bn2)
    s1a, _ = bn_parts(b0.bn1)
    Vfull = (b0.conv2.weight * s2.view(-1, 1, 1, 1))[:, D]                          # 512 x 11 x 3 x 3
    Rch = (b0.conv1.weight * s1a.view(-1, 1, 1, 1))[D]                              # 11 x 256 x 3 x 3
    save["child_writes"] = Vfull.sum((2, 3)).detach().cpu().numpy()
    save["child_reads"] = Rch.detach().cpu().numpy()
    wn = torch.sqrt((Vfull ** 2).sum((0, 2, 3)))                                    # write norm per child
    save["inv_r1"] = (Rch.sum((2, 3)) * wn.view(-1, 1)).sum(0).detach().cpu().numpy()  # per stage-3 channel
    save["D"] = np.array(D)

    np.savez(os.path.join(OUT, "oc_respond" + ("_light" if LIGHT else "") + ".npz"), **save)
    json.dump(res, open(os.path.join(OUT, "oc_respond" + ("_light" if LIGHT else "") + ".json"), "w"), indent=1)
    log("saved", OUT)


if __name__ == "__main__":
    main()
