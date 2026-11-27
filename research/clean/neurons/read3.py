# scale: neurons=a few named ResNet-18 layer4.0 inner channels (+ their layer4.1 readers); net=A, W50R_s0, page-v2 50% build; photos=20,000 ImageNet val (reading + judge only); where=mac
"""Checklist reading of named neurons (research/clean/NEURON-CHECKLIST.md steps 1-5), the same tool for every neuron.

Session log: SESSIONS.md. Usage (through the torch lock):
  curves/.venv/bin/python -I distill/fanout/torchq.py --who s2 -- curves/.venv/bin/python -I neurons/read3.py s2

Per neuron (layer4.0 inner channel, 7 x 7 grid), on READ photos (every 5th val photo, offset 0):
  crops    pos pole = each photo's max cell, neg pole = min cell. Bands: top = ranks 0-15, strong = 16 evenly spaced
           from ranks 16-250. Window = 4 cells centred on the cell, flat grey outside the photo (no mirror).
  votes    the READ: per stage-3 input channel, its share of the pre-activation at the pole cells (3 x 3 kernel, bn1 scale).
           Each voter is shown by the intersection image of its own top windows in ITS network.
  readers  the WRITE: layer4.1 inner channels scored by their conv1 (summed 3 x 3, bn1 scale) . the write (conv2 column,
           bn2 scale). Top 3 turned ON, top 3 turned OFF; each gets a crop sheet (its own top band) + intersection image.
Judge sheets (checklist step 10), on HELD photos (every 5th val photo, offset 1; never used for reading):
  per neuron and pole: 24 target crops (its top band on held photos) + 24 distractors (the top crop of 24 other
  layer4.0 channels of the same network, seeded), shuffled, numbered. Key in out/<tag>/_key/ (judges never read it).
No class labels anywhere."""
import json, os, sys, zlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
CLEAN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG = sys.argv[1] if len(sys.argv) > 1 else "s2"
OUT = os.path.join(CLEAN, "neurons", "out", TAG); KEY = os.path.join(OUT, "_key")
os.makedirs(OUT, exist_ok=True); os.makedirs(KEY, exist_ok=True)
for p in ("distill", "distill/tree", "distill/fanout"):
    sys.path.insert(0, os.path.join(CLEAN, p))
import build as TB, board, wide18  # noqa: E402
torch.set_grad_enabled(False)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
PX = np.load(os.path.join(CLEAN, "hf/imagenet/val_224.npy"), mmap_mode="r")
READ = np.arange(0, 50000, 5)
HELD = np.arange(1, 50000, 5)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ------------------------------------------------------------------ objects (picked by rules written in SESSIONS.md)
NEURONS = {          # name: (network, channel, read writes?, make judge sheets?)
    "X": ("A", 184, True, True),
    "Y": ("W50R", 139, True, True),
    "Z": ("v2", 184, False, False),           # v2 node that holds A ch 184
    "R86": ("W50R", 86, False, False),
    "R219": ("W50R", 219, False, False),
}


def load_nets():
    A = board.r18_A().cpu().eval()
    d = torch.load(os.path.join(CLEAN, "distill/tree/out/cache_r18_724b03416e.pt"), weights_only=False)
    TB.COARSE_PARENT = False; TB.WRITE_PARTNER = False
    g6 = TB.r18_groups(A, d["tr"], 0.5)[0][6]
    S2 = TB.build_r18(A, 0.5, None, st=d["st"], tr=d["tr"], verbose=False).eval()
    dd = torch.load(os.path.join(CLEAN, "neuron/results/coarse/r18/distill_whole/W50R_s0.pt"), map_location="cpu", weights_only=False)
    W = wide18.Student(tuple(dd["widths"])); W.load_state_dict(dd["state"]); W.eval()
    return {"A": A, "W50R": W, "v2": S2}, g6


NETS, G6 = load_nets()
INFO = {}
CH = {}
for k, (net, ch, _, _) in NEURONS.items():
    if net == "v2":
        zi = [i for i, M in enumerate(G6) if ch in M][0]
        INFO[k] = {"v2_group": G6[zi], "v2_index": zi}
        CH[k] = zi
    else:
        CH[k] = ch


def aff(bn):
    return TB._bn_affine(bn)


def readers(m, ch):
    b0, b1 = m.layer4[0], m.layer4[1]
    s2, _ = aff(b0.bn2); w = b0.conv2.weight[:, ch].sum((1, 2)) * s2
    s1, _ = aff(b1.bn1); sc = (b1.conv1.weight.sum((2, 3)) * s1[:, None]) @ w
    o = sc.argsort(descending=True)
    return [int(i) for i in o[:3]] + [int(i) for i in o.flip(0)[:3]], sc


RD = {k: readers(NETS[NEURONS[k][0]], CH[k]) for k in NEURONS if NEURONS[k][2]}
rng = np.random.default_rng(7)
DISTR = {}
for k, (net, ch, _, judge) in NEURONS.items():
    if judge:
        n = NETS[net].layer4[0].conv1.out_channels
        DISTR[k] = [int(c) for c in rng.choice([c for c in range(n) if c != CH[k]], 24, replace=False)]

# ------------------------------------------------------------------ one forward pass per network and photo set
K = 32


def topk_merge(old, vals, idx):
    if old is None:
        return vals, idx
    v = torch.cat([old[0], vals], 1); i = torch.cat([old[1], idx], 1)
    t = v.topk(K, dim=1)
    return t.values, torch.gather(i, 1, t.indices)


def prep(rows):
    return (torch.from_numpy(np.asarray(PX[rows])).permute(0, 3, 1, 2).float() / 255 - MEAN) / STD


def forward(net, rows, want_s3top):
    m = NETS[net].to(DEV)
    units = sorted({CH[k] for k in NEURONS if NEURONS[k][0] == net} | {c for k in DISTR if NEURONS[k][0] == net for c in DISTR[k]})
    rl = sorted({r for k in RD if NEURONS[k][0] == net for r in RD[k][0]})
    maps, rmaps, s3top = [], [], None
    for a in range(0, len(rows), 250):
        x = prep(rows[a:a + 250]).to(DEV)
        s3 = m.layer3(m.layer2(m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(x)))))))
        b0 = m.layer4[0]
        maps.append(b0.bn1(b0.conv1(s3))[:, units].flatten(2).permute(0, 2, 1).cpu())     # (n, 49, u)
        if rl:
            rmaps.append(m.layer4[1].bn1(m.layer4[1].conv1(b0(s3)))[:, rl].flatten(2).permute(0, 2, 1).cpu())
        if want_s3top:
            flat = s3.flatten(2).permute(1, 0, 2).reshape(s3.shape[1], -1).cpu()
            tv = flat.topk(K, dim=1)
            s3top = topk_merge(s3top, tv.values, tv.indices + a * 196)
    m.cpu()
    M = torch.cat(maps)
    out = {"maps": {u: M[..., j] for j, u in enumerate(units)}, "s3top": s3top}
    if rl:
        R = torch.cat(rmaps); out["rmaps"] = {r: R[..., j] for j, r in enumerate(rl)}
    return out


FW = {net: forward(net, READ, True) for net in NETS}
FH = {net: forward(net, HELD, False) for net in {NEURONS[k][0] for k in DISTR}}
print("forward done", flush=True)


# ------------------------------------------------------------------ drawing
def band_cells(Mp, pole, n_top=16, n_strong=16):
    v = Mp if pole == "pos" else -Mp
    best = v.max(1)
    o = best.values.argsort(descending=True)
    sel = torch.cat([o[:n_top], o[torch.linspace(n_top, 250, n_strong).long()]]) if n_strong else o[:n_top]
    return [(int(i), int(best.indices[i]), float(Mp[i, best.indices[i]])) for i in sel]


def window(rows, img_i, cell, grid, cells=4):
    c = 224 // grid; w = cells * c
    im = np.pad(np.asarray(PX[rows[img_i]], np.float32), ((w, w), (w, w), (0, 0)), mode="constant", constant_values=128.0)
    rr, kk = divmod(int(cell), grid)
    cy, cx = int((rr + .5) * c) + w, int((kk + .5) * c) + w
    return im[cy - w // 2:cy + w // 2, cx - w // 2:cx + w // 2]


def edges(a):
    g = a.mean(2); gy, gx = np.gradient(g); return np.hypot(gx, gy)


def to_img(a, S, grey=False):
    if grey:
        a = a / (np.quantile(a, .99) + 1e-6) * 255
        a = np.repeat(np.clip(a, 0, 255)[..., None], 3, 2)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((S, S))


def tile(rows, i, cell, grid, T=120):
    t = Image.new("RGB", (2 * T, T), "white")
    t.paste(to_img(window(rows, i, cell, grid), T), (0, 0))
    whole = Image.fromarray(np.asarray(PX[rows[i]])).resize((T, T)); dw = ImageDraw.Draw(whole)
    c = T / grid; r_, k_ = divmod(cell, grid)
    dw.rectangle([(k_ + .5) * c - 2 * c, (r_ + .5) * c - 2 * c, (k_ + .5) * c + 2 * c, (r_ + .5) * c + 2 * c], outline="red", width=2)
    t.paste(whole, (T, 0))
    return t


def grid_sheet(tiles, labels, cols, path, title, T=120):
    rows_n = (len(tiles) + cols - 1) // cols
    img = Image.new("RGB", (cols * (2 * T + 10), 30 + rows_n * (T + 18)), "white"); dr = ImageDraw.Draw(img)
    dr.text((5, 5), title, fill="black")
    for n, (t, lab) in enumerate(zip(tiles, labels)):
        rr, cc = divmod(n, cols); x0, y0 = cc * (2 * T + 10), 30 + rr * (T + 18)
        img.paste(t, (x0, y0)); dr.text((x0, y0 + T + 2), lab, fill="black")
    img.save(path)


def intersect(rows, cells, grid):
    ws = [window(rows, i, c, grid) for i, c, _ in cells]
    return np.mean(ws, 0), np.mean([edges(w) for w in ws], 0)


def panel(items, path, title):
    S = 140
    img = Image.new("RGB", (2 * S + 300, 30 + len(items) * (S + 8)), "white"); dr = ImageDraw.Draw(img)
    dr.text((5, 5), title, fill="black")
    for r, (lab, cells, grid) in enumerate(items):
        c, e = intersect(READ, cells, grid)
        img.paste(to_img(c, S), (0, 30 + r * (S + 8))); img.paste(to_img(e, S, True), (S + 4, 30 + r * (S + 8)))
        dr.text((2 * S + 12, 30 + r * (S + 8) + S // 2), lab, fill="black")
    img.save(path)


def s3_cells(net, c, n=32):
    v, i = FW[net]["s3top"]; i = i[c]
    return [(int(j // 196), int(j % 196), float(x)) for j, x in zip(i[:n], v[c][:n])]


# ------------------------------------------------------------------ per neuron
res = {"info": INFO, "objects": {k: dict(net=v[0], channel=v[1], index=CH[k]) for k, v in NEURONS.items()}}
for k, (net, _, do_write, _) in NEURONS.items():
    m = NETS[net]; ch = CH[k]; Mp = FW[net]["maps"][ch]
    out = {"fire_rate": round(float((Mp > 0).float().mean()), 3)}
    b0 = m.layer4[0]; s1, _ = aff(b0.bn1); Kr = b0.conv1.weight[ch] * s1[ch]
    for pole in ("pos", "neg"):
        cells = band_cells(Mp, pole)
        grid_sheet([tile(READ, i, c, 7) for i, c, _ in cells], [("top" if n < 16 else "strong") + f" {v:.2f}" for n, (_, _, v) in enumerate(cells)],
                   4, os.path.join(OUT, f"{k}_{pole}_crops.png"), f"neuron {k}  pole {pole}  (4-cell windows, grey outside the photo + whole photo)")
        imgs = sorted({c[0] for c in cells}); pos_of = {im: j for j, im in enumerate(imgs)}
        m.to(DEV)
        s3 = m.layer3(m.layer2(m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(prep(READ[imgs]).to(DEV)))))))).cpu()
        m.cpu()
        V = []
        for im, cell, _ in cells:
            rr, cc = divmod(cell, 7)
            patch = F.pad(s3[pos_of[im]], (1, 1, 1, 1))[:, 2 * rr:2 * rr + 3, 2 * cc:2 * cc + 3]
            V.append((Kr * patch).sum((1, 2)))
        V = torch.stack(V).mean(0)
        up, dn = V.argsort(descending=True)[:5].tolist(), V.argsort()[:5].tolist()
        out[pole] = {"min_max": [round(cells[-1][2], 2), round(cells[0][2], 2)], "vote_sum": round(float(V.sum()), 3),
                     "up_voters": [(int(c), round(float(V[c]), 3)) for c in up], "down_voters": [(int(c), round(float(V[c]), 3)) for c in dn]}
        items = [(f"{k} {pole} pole itself", cells, 7)]
        items += [(f"up voter s3:{c} ({float(V[c]):+.2f})", s3_cells(net, c), 14) for c in up[:3]]
        items += [(f"down voter s3:{c} ({float(V[c]):+.2f})", s3_cells(net, c), 14) for c in dn[:3]]
        panel(items, os.path.join(OUT, f"{k}_{pole}_intersect_voters.png"), f"neuron {k} {pole}: colour mean | edge mean")
    if do_write:
        rl, sc = RD[k]
        out["readers"] = [(r, round(float(sc[r]), 3), "ON" if j < 3 else "OFF") for j, r in enumerate(rl)]
        tiles, labels = [], []
        for j, r in enumerate(rl):
            cells = band_cells(FW[net]["rmaps"][r], "pos", 6, 0)
            tiles += [tile(READ, i, c, 7) for i, c, _ in cells]
            labels += [f"reader {j + 1} ({'ON' if j < 3 else 'OFF'} {float(sc[r]):+.2f})  {v:.2f}" for _, _, v in cells]
        grid_sheet(tiles, labels, 6, os.path.join(OUT, f"{k}_write_reader_crops.png"),
                   f"neuron {k} WRITE: one row per next-layer reader (rows 1-3 turned ON, rows 4-6 turned OFF), its 6 top windows")
        panel([(f"reader {j + 1} {'ON' if j < 3 else 'OFF'} ({float(sc[r]):+.2f})", band_cells(FW[net]["rmaps"][r], "pos", 32, 0), 7) for j, r in enumerate(rl)],
              os.path.join(OUT, f"{k}_write_readers.png"), f"neuron {k} WRITE: readers' intersection images")
    res[k] = out

# ------------------------------------------------------------------ judge sheets (held photos)
key = {}
for k in DISTR:
    net = NEURONS[k][0]; H = FH[net]["maps"]
    for pole in ("pos", "neg"):
        tgt = band_cells(H[CH[k]], pole, 24, 0)
        dis = [band_cells(H[c], "pos", 1, 0)[0] for c in DISTR[k]]
        allc = [(c, 1) for c in tgt] + [(c, 0) for c in dis]
        order = np.random.default_rng(zlib.crc32(f"{k}{pole}".encode())).permutation(len(allc))
        allc = [allc[i] for i in order]
        sid = f"J_{k}_{pole}"
        grid_sheet([tile(HELD, i, c, 7) for (i, c, _), _ in allc], [f"#{n + 1}" for n in range(len(allc))], 6,
                   os.path.join(OUT, f"{sid}.png"), f"sheet {sid[2:].replace('_', ' ')} : 48 numbered tiles")
        key[sid] = [n + 1 for n, (_, t) in enumerate(allc) if t]
json.dump(key, open(os.path.join(KEY, "key.json"), "w"))
json.dump(res, open(os.path.join(OUT, "read.json"), "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("fire_rate", "readers")} | {p: v[p]["min_max"] for p in ("pos", "neg")}
                  for k, v in res.items() if k in NEURONS}, indent=0))
print("info", INFO)
