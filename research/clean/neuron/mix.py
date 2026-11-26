"""Phase 1 of plan part 2: what each child mixes in besides its parent (NEURON-LOG N2).

Child = a stage-4 block-1 inner channel s4b1i:j. Its pre-activation u_j reads the block-0 stream s4b0 through conv1 + bn1.
Exact split, against the fill (as the map tool does): u_j(pos) = base_j + sum_c part_c(pos), with
    part_c = scale_j * (W1[j, c] conv (h_c - mean_c)),   base_j = u_j with every s4b0 channel at its mean.
A firing = one photo at the child's peak position. Strong firings = the 500 strongest fit photos (and 500 test photos).
Subcircuits = k-means on the firings' input profiles (parts, divided by the firing's total positive part), k from 2 to 5
by silhouette, fixed seed. Test firings go to the nearest centre.
Effect of a firing on the logits = KL(N || N with the child at its mean), per photo.
"""
import json
import os

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from absorb import net, stage4, DEV, BATCH, OUT

torch.set_grad_enabled(False)
CHILDREN = {243: 245, 73: 245, 377: 245, 317: 418, 408: 418}
NSTRONG = 500
SHEETS = os.path.join(OUT, "mix_sheets")
os.makedirs(SHEETS, exist_ok=True)


def card_names():
    import glob
    names = {}
    for f in glob.glob(os.path.join(net.HERE, "cards2", "*.jsonl")):
        for line in open(f):
            try:
                c = json.loads(line)
            except Exception:
                continue
            names[c["node"]] = c["distinction"][:70]
    return names


def main():
    m = net.load().to(DEV)
    ph = net.Photos("read")
    lab = ph.lab
    pos_in_class = np.zeros(len(lab), int)
    for c in range(1000):
        w = np.where(lab == c)[0]
        pos_in_class[w] = np.arange(len(w))
    fit = pos_in_class % 2 == 0
    cache = np.load(os.path.join(OUT, "absorb_cache.npz"))
    inner_means = torch.tensor(cache["means"]).to(DEV)
    h_mean = net.block_means(os.path.join(net.STORE, "read"))[4][0].to(DEV)  # s4b0 channel means, reading photos
    b1 = m.layer4[1]
    sc = (b1.bn1.weight / torch.sqrt(b1.bn1.running_var + b1.bn1.eps))
    kids = list(CHILDREN)
    W1 = b1.conv1.weight[kids]  # (5, 512, 3, 3)
    mean_map = h_mean.view(1, -1, 1, 1).expand(1, 512, 7, 7).contiguous()
    mean_pad = torch.nn.functional.pad(mean_map, (1, 1, 1, 1))
    u_fill = b1.bn1(b1.conv1(mean_map))[0]  # (512, 7, 7): each inner pre-activation with s4b0 at its mean

    # Pass 1: every photo, each child's peak value and position, and its most negative value.
    N = len(lab)
    peak = np.zeros((N, len(kids)), np.float32)
    ppos = np.zeros((N, len(kids)), int)
    low = np.zeros((N, len(kids)), np.float32)
    lpos = np.zeros((N, len(kids)), int)
    for i in range(0, N, BATCH):
        x = ph.tensor(np.arange(i, min(i + BATCH, N))).float().to(DEV)
        h3 = m.layer3(m.layer2(m.layer1(net.stem(m, x))))
        _, u, _ = stage4(m, h3)
        uk = u[:, kids].flatten(2)  # (B, 5, 49)
        mx, am = uk.max(2)
        mn, an = uk.min(2)
        sl = slice(i, i + len(x))
        peak[sl], ppos[sl], low[sl], lpos[sl] = mx.cpu().numpy(), am.cpu().numpy(), mn.cpu().numpy(), an.cpu().numpy()
    print("pass 1 done", flush=True)

    names = card_names()
    out = {}
    for ki, j in enumerate(kids):
        p = CHILDREN[j]
        sets = {}
        for half, mask in (("fit", fit), ("test", ~fit)):
            idx = np.where(mask)[0]
            sets[half] = idx[np.argsort(-peak[idx, ki])[:NSTRONG]]
        sets["neg"] = np.where(fit)[0][np.argsort(low[fit, ki])[:200]]
        prof, eff, val = {}, {}, {}
        for half, idx in sets.items():
            P = np.zeros((len(idx), 512))
            V = np.zeros(len(idx))
            E = np.zeros(len(idx))
            B0 = np.zeros(len(idx))
            pp = (lpos if half == "neg" else ppos)[idx, ki]
            for s in range(0, len(idx), BATCH):
                ii = idx[s:s + BATCH]
                x = ph.tensor(ii).float().to(DEV)
                h3 = m.layer3(m.layer2(m.layer1(net.stem(m, x))))
                h40, u, h41 = stage4(m, h3)
                # the conv pads h with 0, so the fill is the mean inside the map and 0 in the padding
                dh = torch.nn.functional.pad(h40, (1, 1, 1, 1)) - mean_pad
                r, q = torch.as_tensor(pp[s:s + BATCH] // 7), torch.as_tensor(pp[s:s + BATCH] % 7)
                win = torch.stack([dh[b, :, r[b]:r[b] + 3, q[b]:q[b] + 3] for b in range(len(ii))])  # (B, 512, 3, 3)
                part = (win * W1[ki].unsqueeze(0)).sum((2, 3)) * sc[j]  # (B, 512)
                B0[s:s + len(ii)] = u_fill[j].flatten()[torch.as_tensor(pp[s:s + BATCH])].cpu().numpy()
                P[s:s + len(ii)] = part.cpu().numpy()
                at = torch.arange(len(ii))
                V[s:s + len(ii)] = u[:, j].flatten(1)[at, torch.as_tensor(pp[s:s + BATCH])].cpu().numpy()
                if half != "neg":
                    _, _, hd = stage4(m, h3, drop=[j], means=inner_means)
                    yN = m.fc(h41.mean((2, 3)))
                    yD = m.fc(hd.mean((2, 3)))
                    lpN, lpD = torch.log_softmax(yN, 1), torch.log_softmax(yD, 1)
                    E[s:s + len(ii)] = (lpN.exp() * (lpN - lpD)).sum(1).cpu().numpy()
            prof[half], val[half], eff[half] = P, V, E
            if half == "fit":
                base_fit = B0
        # exact check: value = base (all s4b0 channels at their means) + the sum of the parts
        miss = float(np.max(np.abs(val["fit"] - prof["fit"].sum(1) - base_fit)) / np.mean(np.abs(val["fit"])))
        if miss > 2e-3:
            raise SystemExit(f"sum check failed on child {j}: relative {miss:.2e}")

        def norm(P):
            pos = np.clip(P, 0, None).sum(1, keepdims=True)
            return P / np.maximum(pos, 1e-9)
        Xf, Xt = norm(prof["fit"]), norm(prof["test"])
        best = None
        for k in range(2, 6):
            km = KMeans(k, n_init=10, random_state=0).fit(Xf)
            s_ = silhouette_score(Xf, km.labels_)
            if best is None or s_ > best[0]:
                best = (s_, k, km)
        sil, k, km = best
        lt = km.predict(Xt)
        subs = []
        for c in range(k):
            mf = km.labels_ == c
            mt = lt == c
            mp = prof["fit"][mf].mean(0)
            top = np.argsort(-mp)[:10]
            posum = np.clip(prof["fit"][mf], 0, None).sum(1).mean()
            cls = np.bincount(lab[sets["fit"][mf]], minlength=1000)
            topc = np.argsort(-cls)[:6]
            subs.append(dict(
                id=c, n_fit=int(mf.sum()), n_test=int(mt.sum()),
                effect_share_test=float(eff["test"][mt].sum() / max(eff["test"].sum(), 1e-12)),
                parent_share=float(max(mp[p], 0) / max(posum, 1e-9)),
                parent_rank=int(np.where(np.argsort(-mp) == p)[0][0]) + 1,
                top10=[int(t) for t in top],
                top3=[(f"s4b0:{t}", round(float(mp[t]), 3), names.get(f"s4b0:{t}", "unread")) for t in top[:3]],
                classes=[(net.class_names()[c_], int(cls[c_])) for c_ in topc],
                value_mean=float(val["fit"][mf].mean())))
            # photo sheet: 12 strongest fit firings of this subcircuit
            ii = sets["fit"][mf][np.argsort(-val["fit"][mf])[:12]]
            raw = ph.raw(ii)
            sheet = Image.new("RGB", (6 * 112, 2 * 112), "white")
            for t, im in enumerate(raw):
                sheet.paste(Image.fromarray(np.asarray(im)).resize((112, 112)), ((t % 6) * 112, (t // 6) * 112))
            sheet.save(os.path.join(SHEETS, f"child{j}_sub{c}.png"))
        # overlap of top-10 inputs between subcircuits
        ov = [len(set(a["top10"]) & set(b["top10"])) / 10 for x, a in enumerate(subs) for b in subs[x + 1:]]
        mn = prof["neg"].mean(0)
        negtop = np.argsort(mn)[:5]
        out[j] = dict(parent=p, sum_check_rel=miss, silhouette=float(sil), k=k, subcircuits=subs,
                      overlap_median=float(np.median(ov)) if ov else None,
                      neg_inputs=[(f"s4b0:{t}", round(float(mn[t]), 3), names.get(f"s4b0:{t}", "unread")) for t in negtop],
                      neg_classes=[net.class_names()[c_] for c_ in np.argsort(-np.bincount(lab[sets["neg"]], minlength=1000))[:6]],
                      parent_share_all=float(np.clip(prof["fit"][:, p], 0, None).sum() / np.clip(prof["fit"], 0, None).sum()))
        print(json.dumps({"child": j, **{k_: v for k_, v in out[j].items() if k_ != "subcircuits"}}), flush=True)
        for s_ in subs:
            print("   ", json.dumps(s_), flush=True)
    # siblings: each child's value on its sibling's 100 strongest fit photos, as a percentile of its own values
    sib = {}
    allv = {ki: np.sort(peak[fit, ki]) for ki in range(len(kids))}
    for a, ja in enumerate(kids):
        for b, jb in enumerate(kids):
            if a != b and CHILDREN[ja] == CHILDREN[jb]:
                top_b = np.where(fit)[0][np.argsort(-peak[fit, b])[:100]]
                pct = np.searchsorted(allv[a], peak[top_b, a]) / len(allv[a])
                sib[f"{ja}_on_{jb}"] = float(np.median(pct))
    out["siblings"] = sib
    print("siblings", sib, flush=True)
    json.dump(out, open(os.path.join(OUT, "mix.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
