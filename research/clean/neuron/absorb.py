"""Step 1 of neuron coarsening, picture A: the parent absorbs (plan i-ve-thought-about-our-stateful-feigenbaum.md).

Object, in Shonu's words: a child disappears, and the parent takes up some of the error.

Network: torchvision ResNet-18 (IMAGENET1K_V1).
  parent    a stage-4 block-0 stream channel s4b0:p.
  children  stage-4 block-1 inner channels s4b1i:j that read p (map children from the read cards, or direct-edge children).
  readout   the fc layer. p reaches it through the skip: s4b1:p = relu(s4b0:p + residual).

Recipe: remove the children (each filled with its own mean, at every position).
Absorb (calculated, no training): change only the readout of the parent's channel, column p of fc and the bias:
    y_abs(x) = y_drop(x) + a + b * g_p(x),   g_p = pooled s4b1:p in the dropped network.
  closed form: least squares of the centred logit gap y_N - y_drop on [1, g_p]   (a squared-logit fit: label OPT-D2)
  KL optimum of the same form: E_P KL(softmax y_N || softmax y_abs) over (a, b). Convex, so one global optimum.
  whole readout: all of fc and the bias refit on KL. Convex too. It shows what any readout change could absorb.

Photos: the reading photos only (a lock closes the sealed ones). Inside each class, even positions fit, odd positions test.
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
R18 = os.path.join(HERE, "..", "curves", "map_agents", "resnet18")
sys.path.insert(0, R18)
import net  # noqa: E402

torch.set_grad_enabled(False)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

# Parents and recipes. Map children come from the read cards (cards2/*.jsonl, field "inputs").
PARENTS = {
    245: dict(name="small coloured bird", map_children=[243, 377, 73]),
    418: dict(name="long muzzle profile", map_children=[408, 377]),
    237: dict(name="slim glossy contrast body (read negatively)", map_children=[243, 98, 500, 383, 58, 189]),
}
TOPK_EDGE = 10  # second recipe: the 10 inner channels with the largest direct positive edge from p
BATCH = 125


def stage4(m, h3, drop=None, means=None):
    """Stage 4 by hand. drop: list of block-1 inner channels, filled with means. Returns block-0 h, block-1 inner pre-act, final h."""
    b0, b1 = m.layer4[0], m.layer4[1]
    p0 = net.block_parts(b0, h3)
    h40 = p0["h"]
    u = b1.bn1(b1.conv1(h40))
    inner = torch.relu(u)
    if drop:
        inner = inner.clone()
        inner[:, drop] = means[drop].view(1, -1, 1, 1)
    z = b1.bn2(b1.conv2(inner)) + h40
    return h40, u, torch.relu(z)


def main():
    t0 = time.time()
    m = net.load().to(DEV)
    ph = net.Photos("read")
    lab = ph.lab
    pos_in_class = np.zeros(len(lab), int)
    for c in range(1000):
        w = np.where(lab == c)[0]
        pos_in_class[w] = np.arange(len(w))
    fit = pos_in_class % 2 == 0

    b1 = m.layer4[1]
    s1 = (b1.bn1.weight / torch.sqrt(b1.bn1.running_var + b1.bn1.eps)).view(1, -1, 1, 1)

    # Pass A: block-1 inner channel means, on 4000 fit photos (every class, 4 each).
    idxA = np.where(fit & (pos_in_class < 8))[0]
    acc = torch.zeros(512, device=DEV)
    n = 0
    for i in range(0, len(idxA), BATCH):
        x = ph.tensor(idxA[i:i + BATCH]).to(DEV)
        h3 = m.layer3(m.layer2(m.layer1(net.stem(m, x))))
        _, u, _ = stage4(m, h3)
        acc += torch.relu(u).mean((2, 3)).sum(0)
        n += x.shape[0]
    means = (acc / n)
    print(f"pass A done, {n} photos, {time.time() - t0:.0f}s", flush=True)

    # Recipes: map children, and top direct-edge children (picked on pass B's fit photos would leak nothing to test, but
    # it needs edges first; so edges come from pass A photos below, in a short extra loop).
    edge_sum = {p: torch.zeros(512, device=DEV) for p in PARENTS}
    edge_cnt = {p: torch.zeros(512, device=DEV) for p in PARENTS}
    pre_sum = torch.zeros(512, device=DEV)
    for i in range(0, len(idxA), BATCH):
        x = ph.tensor(idxA[i:i + BATCH]).to(DEV)
        h3 = m.layer3(m.layer2(m.layer1(net.stem(m, x))))
        h40, u, _ = stage4(m, h3)
        on = (u > 0).float()
        pre_sum += (u * on).sum((0, 2, 3))
        for p in PARENTS:
            w = b1.conv1.weight[:, p:p + 1]  # (512, 1, 3, 3)
            c = torch.nn.functional.conv2d(h40[:, p:p + 1], w, padding=1) * s1  # p's exact part of every inner pre-act
            edge_sum[p] += (c * on).sum((0, 2, 3))
            edge_cnt[p] += on.sum((0, 2, 3))
    edges = {}
    for p in PARENTS:
        e = (edge_sum[p] / edge_cnt[p].clamp(min=1)).cpu().numpy()  # mean part of p in j's pre-act, where j is on
        share = (edge_sum[p] / pre_sum.clamp(min=1e-9)).cpu().numpy()  # share of j's on pre-act that p gives
        edges[p] = dict(mean=e, share=share)
    recipes = {}
    for p, d in PARENTS.items():
        top = np.argsort(-edges[p]["mean"])[:TOPK_EDGE].tolist()
        recipes[(p, "map")] = d["map_children"]
        recipes[(p, "edge")] = top
    print(f"edges done, {time.time() - t0:.0f}s", flush=True)

    # Pass B: every reading photo. Pooled final stream for N and for each recipe; pooled parent s4b0:p.
    N = len(lab)
    gN = np.zeros((N, 512), np.float32)
    gD = {k: np.zeros((N, 512), np.float32) for k in recipes}
    par = np.zeros((N, len(PARENTS)), np.float32)
    plist = list(PARENTS)
    for i in range(0, N, BATCH):
        idx = np.arange(i, min(i + BATCH, N))
        x = ph.tensor(idx).to(DEV)
        h3 = m.layer3(m.layer2(m.layer1(net.stem(m, x))))
        h40, _, h41 = stage4(m, h3)
        gN[idx] = h41.mean((2, 3)).cpu().numpy()
        par[idx] = h40[:, plist].mean((2, 3)).cpu().numpy()
        for k, ch in recipes.items():
            _, _, hd = stage4(m, h3, drop=ch, means=means)
            gD[k][idx] = hd.mean((2, 3)).cpu().numpy()
        if (i // BATCH) % 40 == 0:
            print(f"  pass B {i}/{N}, {time.time() - t0:.0f}s", flush=True)

    np.savez_compressed(os.path.join(OUT, "absorb_cache.npz"), gN=gN, par=par, lab=lab, fit=fit,
                        **{f"gD_{p}_{r}": v for (p, r), v in gD.items()},
                        fcW=m.fc.weight.cpu().numpy(), fcb=m.fc.bias.cpu().numpy(), means=means.cpu().numpy())
    json.dump(dict(parents={str(p): d for p, d in PARENTS.items()},
                   recipes={f"{p}_{r}": v for (p, r), v in recipes.items()},
                   edges={str(p): dict(mean=e["mean"].tolist(), share=e["share"].tolist()) for p, e in edges.items()}),
              open(os.path.join(OUT, "absorb_edges.json"), "w"))
    print(f"saved, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
