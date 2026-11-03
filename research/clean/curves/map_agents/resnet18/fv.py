# serves: H-map
"""Feature visualization for the map agents. Fourier image, decorrelated colour, and transformation robustness (Olah et al. 2017).

Free optimization can find routes real photos rarely use, the essay's "reward hacking". So --natural adds the inputs real photos use for this
node, for the first 64 steps only. --through PARENT adds that one input instead. Both views go side by side when both are asked for.

The objective is the node's value before its ReLU at the centre position. Batched, so a slice can be computed once.
Usage: fv.py --nodes s4b1:179,s4b1i:12 [--steps 256] [--gpu]      or from ev2.py: fv NODE [--natural] [--through NODE]"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from net import HERE, STORE, Photos, forward, load, stages  # noqa: E402

OUT = os.path.join(HERE, "sheets2", "fv")
COLOR = torch.tensor([[.26, .09, .02], [.27, .00, -.05], [.27, -.09, .03]])
COLOR = COLOR / COLOR.norm(dim=0).max()
SIZE = 224
MEAN, STD = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1), torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)


def image_of(spec, freq):
    im = torch.fft.irfft2(torch.complex(spec[..., 0], spec[..., 1]) * freq, s=(SIZE, SIZE)) * 4
    im = (COLOR.to(im) @ im.permute(0, 2, 3, 1)[..., None])[..., 0].permute(0, 3, 1, 2)
    return torch.sigmoid(im)


def jitter(x, rng):
    """Pad, shift, scale, rotate, shift. One random transform per image."""
    n = x.shape[0]
    x = F.pad(x, (12, 12, 12, 12), value=.5)
    ang = torch.as_tensor(rng.uniform(-5, 5, n) * np.pi / 180, dtype=x.dtype, device=x.device)
    sc = torch.as_tensor(rng.choice([1, .975, 1.025, .95, 1.05], n), dtype=x.dtype, device=x.device)
    tx = torch.as_tensor(rng.uniform(-.06, .06, (n, 2)), dtype=x.dtype, device=x.device)
    theta = torch.zeros(n, 2, 3, dtype=x.dtype, device=x.device)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = torch.cos(ang) / sc, -torch.sin(ang) / sc, torch.sin(ang) / sc, torch.cos(ang) / sc
    theta[:, :, 2] = tx
    g = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, g, padding_mode="border", align_corners=False)[:, :, 12:-12, 12:-12]


def value(m, o, n):
    """The node's value before its ReLU, over positions."""
    from ev2 import below
    if n["s"] == 0:
        return o[(0, 0)]["h"][:, n["c"]]
    if n["inner"]:
        bl = stages(m)[n["s"]][n["b"]]
        return bl.bn1(bl.conv1(o[below(n)]["h"]))[:, n["c"]]
    return o[(n["s"], n["b"])]["z"][:, n["c"]]


def natural_route(m, n, count=6):
    """The inputs that real photos use most for this node. Evidence for the optimizer only."""
    from ev2 import exact_inputs, parse, top_photos
    if n["s"] == 0:
        return []
    idx, pos = top_photos(n, "top", 32)
    D, names, _, _ = exact_inputs(m, Photos("read").tensor(idx), n, pos)
    me = D.mean(0)
    return [parse(names[j].replace(" (skip)", "")) for j in np.argsort(-me)[:count] if me[j] > 0]


def render(nodes, helpers, steps=256, device="cpu", seed=0):
    """nodes: list of node dicts. helpers: for each node, a list of nodes pushed for the first 64 steps. Returns uint8 images."""
    dev = torch.device(device)
    m = load().to(dev).float()
    n = len(nodes)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    fy = torch.fft.fftfreq(SIZE)[:, None]
    fx = torch.fft.rfftfreq(SIZE)[None, :]
    freq = (1 / torch.maximum(torch.sqrt(fx * fx + fy * fy), torch.tensor(1 / SIZE))).to(dev)
    spec = (torch.randn(n, 3, SIZE, SIZE // 2 + 1, 2, device=dev) * .01).requires_grad_(True)
    opt = torch.optim.Adam([spec], lr=.05)
    mean, std = MEAN.to(dev), STD.to(dev)
    for step in range(steps):
        opt.zero_grad()
        im = jitter(image_of(spec, freq), rng)
        _, o = forward(m, (im - mean) / std)
        loss = 0.0
        for i, node in enumerate(nodes):
            f = value(m, o, node)[i]
            loss = loss - f[f.shape[0] // 2, f.shape[1] // 2]
            if step < 64:
                for h in helpers[i]:
                    g = value(m, o, h)[i]
                    loss = loss - g.mean() * .5
        loss.backward()
        opt.step()
    with torch.no_grad():
        final = image_of(spec, freq).permute(0, 2, 3, 1).cpu().numpy()
    return (final * 255).astype(np.uint8)


def show(text, natural, through):
    from PIL import Image
    from ev2 import name, parse
    n = parse(text)
    m = load()
    views = {"free": []}
    if natural:
        views["natural"] = natural_route(m, n)
    if through:
        views["through"] = [parse(through)]
    os.makedirs(OUT, exist_ok=True)
    out = []
    for view, helpers in views.items():
        p = os.path.join(OUT, f"{name(n).replace(':', '_')}_{view}.png")
        if not os.path.exists(p) or view == "through":
            Image.fromarray(render([n], [helpers], steps=256)[0]).save(p)
        out.append(p)
    print("\n".join(out))


if __name__ == "__main__":
    from PIL import Image
    from ev2 import name, parse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = "mps" if a.gpu else "cpu"
    nodes = [parse(t) for t in a.nodes.split(",")]
    base = load()
    os.makedirs(OUT, exist_ok=True)
    for view in ("free", "natural"):
        for i in range(0, len(nodes), a.batch):
            part = [x for x in nodes[i:i + a.batch] if not os.path.exists(os.path.join(OUT, f"{name(x).replace(':', '_')}_{view}.png"))]
            if not part:
                continue
            helpers = [[] if view == "free" else natural_route(base, x) for x in part]
            images = render(part, helpers, a.steps, device)
            for x, im in zip(part, images):
                Image.fromarray(im).save(os.path.join(OUT, f"{name(x).replace(':', '_')}_{view}.png"))
            print(view, i, flush=True)
