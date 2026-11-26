"""Pair study: where is the Small neuron s on while its counterpart p is off, and what in Base fires there?

14 pairs (the 14 strongest of kids.json, including the family 1711, 151, 799, 377). Photos: the 4,000 of pos.py.
Rows = photo x position. "s on, p off" rows: the top 150 rows by s's pre-activation among rows where p's
pre-activation is below 0. For those rows: every Base stage-4 neuron's post-GELU output, z-scored against all rows.
The top 8 Base neurons per pair, with their child rank under p (kids_scores, blocks 1-2).
Photo sheets: per pair, row 1 = 8 crops where s is on and p off; row 2 = 8 crops of p's own top rows.
"""
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from cache import DEV, OUT, hook_stage4, load, net
from pos import photos

torch.set_grad_enabled(False)
K = 150


def main():
    kj = json.load(open(os.path.join(OUT, "kids.json")))[-14:]
    ksc = np.load(os.path.join(OUT, "kids_scores.npy"))[-14:]
    ph = net.Photos("read")
    tr, te = photos(ph)
    allix = np.sort(np.concatenate([tr, te]))
    ms, mb = load("small"), load("base")
    st_s, st_b = {}, {}
    hook_stage4(ms, st_s)
    G = {}
    for b, blk in enumerate(mb.features[7]):
        blk.block[3].register_forward_hook(lambda m, i, o, b=b: st_b.__setitem__(b, o))
        blk.block[4].register_forward_hook(lambda m, i, o, b=b: G.__setitem__(b, o))
    S_ix = torch.tensor([q["s"] for q in kj], device=DEV)
    P_ix = torch.tensor([q["p"] for q in kj], device=DEV)
    nP = len(kj)
    top_v = torch.full((nP, K), -1e9)
    top_id = torch.zeros(nP, K, dtype=torch.long)
    top_g = torch.zeros(nP, K, 12288)
    ptop_v = torch.full((nP, K), -1e9)
    ptop_id = torch.zeros(nP, K, dtype=torch.long)
    gs, gss, gn = torch.zeros(12288, dtype=torch.float64), torch.zeros(12288, dtype=torch.float64), 0
    for k in range(0, len(allix), 50):
        ii = allix[k:k + 50]
        x = ph.tensor(ii).to(DEV)
        ms(x), mb(x)
        s = st_s[0].reshape(-1, 3072)[:, S_ix].float()
        p = st_b[0].reshape(-1, 4096)[:, P_ix].float()
        g = torch.cat([G[b].reshape(-1, 4096) for b in range(3)], 1).float()
        gs += g.sum(0).cpu().double()
        gss += (g * g).sum(0).cpu().double()
        gn += g.shape[0]
        rid = torch.arange(k * 49, k * 49 + g.shape[0])  # row id = photo order * 49 + position
        cand = torch.where(p < 0, s, torch.full_like(s, -1e9)).cpu()
        for i in range(nP):
            v = torch.cat([top_v[i], cand[:, i]])
            ids = torch.cat([top_id[i], rid])
            o = torch.topk(v, K).indices
            newg = torch.cat([top_g[i], torch.zeros(0, 12288)])
            keep_old = o[o < K]
            keep_new = o[o >= K] - K
            gnew = g[keep_new.to(DEV)].cpu() if len(keep_new) else torch.zeros(0, 12288)
            top_g[i] = torch.cat([newg[keep_old], gnew])
            top_v[i] = torch.cat([v[keep_old], v[keep_new + K]])
            top_id[i] = torch.cat([ids[keep_old], ids[keep_new + K]])
            pv = torch.cat([ptop_v[i], p[:, i].cpu()])
            pid = torch.cat([ptop_id[i], rid])
            o2 = torch.topk(pv, K).indices
            ptop_v[i], ptop_id[i] = pv[o2], pid[o2]
    mu = gs / gn
    sd = (gss / gn - mu ** 2).clamp_min(1e-12).sqrt()
    out = []
    os.makedirs(os.path.join(OUT, "sheets"), exist_ok=True)
    for i, q in enumerate(kj):
        zz = ((top_g[i].double().mean(0) - mu) / sd).numpy()
        zz[q["p"]] = zz[q["p"]]
        best = np.argsort(-zz)[:8]
        rows = []
        for f in best:
            blk, c = int(f // 4096), int(f % 4096)
            rank = int((ksc[i] > ksc[i][f - 4096]).sum()) if f >= 4096 else None
            rows.append(dict(block=blk, c=c, z=float(zz[f]), child_rank=rank))
        out.append(dict(s=q["s"], p=q["p"], p_z_there=float(zz[q["p"]]), top=rows,
                        s_on_p_off_value_median=float(top_v[i].median())))
        print(f"s{q['s']}~p{q['p']}: p z {zz[q['p']]:+.2f}; top Base there: "
              + ", ".join(f"b{r['block']}:{r['c']} z{r['z']:.1f} cr{r['child_rank']}" for r in rows[:5]), flush=True)
        # sheet
        canvas = Image.new("RGB", (8 * 100, 2 * 100 + 20), "white")
        for row, ids in enumerate([top_id[i], ptop_id[i]]):
            seen, col = set(), 0
            for rid in ids.tolist():
                ph_i, pos = allix[rid // 49], rid % 49
                if ph_i in seen:
                    continue
                seen.add(ph_i)
                img = Image.fromarray(np.asarray(ph.raw([ph_i])[0]))
                cy, cx = (pos // 7) * 32 + 16, (pos % 7) * 32 + 16
                box = (max(cx - 48, 0), max(cy - 48, 0), min(cx + 48, 224), min(cy + 48, 224))
                canvas.paste(img.crop(box).resize((96, 96)), (col * 100, row * 100 + 20))
                col += 1
                if col == 8:
                    break
        ImageDraw.Draw(canvas).text((4, 4), f"top: Small s{q['s']} on, Base p{q['p']} off   bottom: p{q['p']} top", fill="black")
        canvas.save(os.path.join(OUT, "sheets", f"s{q['s']}_p{q['p']}.png"))
    json.dump(out, open(os.path.join(OUT, "sheet.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
