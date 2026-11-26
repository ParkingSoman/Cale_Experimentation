"""N5 step 3: trained reference for a shrunk tree. Same form as the build: only the kept tree members' reads, biases and
writes, the tree blocks' output biases, and the roots' writes may change. Trained by KL to Base on the calibration photos
(120 group + 120 other fit photos, the build's own). Learning rate and epoch picked on validation photos (other fit
photos of the group + 60 other fit photos). Scored on the same held-out test photos as shrink.py: group KL, coarse KL,
other KL. Starts: the parent build and the drop_goal build (shrink_state_<group>_<var>_f<f>.pt).
Usage: python ref.py GROUP [f]
"""
import copy
import json
import os
import sys
import time

import numpy as np
import torch

from build_tree import photos
from common import DEV, OUT, blocks, coarse_kl_rows, groups, kl_rows, load, net, split

EPOCHS, BS = 15, 20


def main():
    torch.set_grad_enabled(True)  # cache.py turns gradients off when imported
    name = sys.argv[1]
    f = sys.argv[2] if len(sys.argv) > 2 else "0.25"
    grp = groups()[name]
    T = json.load(open(os.path.join(OUT, "trees2.json")))[name]
    te_g, te_o = np.array(T["test_group"]), np.array(T["test_other"])
    ph = net.Photos("read")
    fit = split(ph.lab)
    base = load("base")
    for p in base.parameters():
        p.requires_grad_(False)
    st0 = torch.load(os.path.join(OUT, f"shrink_state_{name}_parent_f{f}.pt"), weights_only=True)
    cal = np.array(st0["cal"])
    others = np.setdiff1d(np.where(fit)[0], np.concatenate([np.where(np.isin(ph.lab, grp))[0], cal]))
    val = np.sort(np.concatenate([photos(ph, grp, fit, 13, max(1, -(-60 // len(grp))), 60),
                                  np.random.default_rng(9).choice(others, 60, replace=False)]))
    with torch.no_grad():
        teach = {k: torch.cat([base(ph.tensor(ids[a:a + 40]).to(DEV)) for a in range(0, len(ids), 40)])
                 for k, ids in (("cal", cal), ("val", val), ("g", te_g), ("o", te_o))}

    def score(m, ids, ref):
        kl, ck = [], []
        with torch.no_grad():
            for a in range(0, len(ids), 40):
                y = m(ph.tensor(ids[a:a + 40]).to(DEV))
                kl.append(kl_rows(ref[a:a + 40], y).cpu())
                ck.append(coarse_kl_rows(ref[a:a + 40], y, grp).cpu())
        return torch.cat(kl).numpy(), torch.cat(ck).numpy()

    res = {}
    for start in ("parent", "drop_goal"):
        st = torch.load(os.path.join(OUT, f"shrink_state_{name}_{start}_f{f}.pt"), weights_only=True)
        best_overall = None
        for lr in (1e-4, 1e-3):
            m = copy.deepcopy(base)
            BL = blocks(m)
            params = []
            for b, (W1, b1, W2, b2) in st["blocks"].items():
                b = int(b)
                blk = BL[b]
                with torch.no_grad():
                    blk.block[3].weight.copy_(W1)
                    blk.block[3].bias.copy_(b1)
                    blk.block[5].weight.copy_(W2)
                    blk.block[5].bias.copy_(b2)
                h = blk.block[3].out_features
                mask_r = torch.zeros(h, device=DEV)
                if b == st["root_block"]:
                    mask_r[torch.tensor(st["roots"])] = 1  # roots: write only
                    trainable = [(blk.block[5].weight, mask_r[None, :]), (blk.block[5].bias, None)]
                else:
                    if len(st["kept"][b]):
                        mask_r[torch.tensor(st["kept"][b])] = 1
                    trainable = [(blk.block[3].weight, mask_r[:, None]), (blk.block[3].bias, mask_r),
                                 (blk.block[5].weight, mask_r[None, :]), (blk.block[5].bias, None)]
                for p, mk in trainable:
                    p.requires_grad_(True)
                    if mk is not None:
                        p.register_hook(lambda g, mk=mk: g * mk)
                    params.append(p)
            opt = torch.optim.Adam(params, lr=lr)
            snap = lambda: {i: p.detach().clone() for i, p in enumerate(params)}
            v0 = score(m, val, teach["val"])[0].mean()
            best = (float(v0), 0, snap())
            t0 = time.time()
            for ep in range(1, EPOCHS + 1):
                perm = np.random.default_rng(ep).permutation(len(cal))
                for a in range(0, len(cal), BS):
                    ii = perm[a:a + BS]
                    y = m(ph.tensor(cal[ii]).to(DEV))
                    loss = kl_rows(teach["cal"][torch.as_tensor(ii, device=DEV)], y).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                v = float(score(m, val, teach["val"])[0].mean())
                if v < best[0]:
                    best = (v, ep, snap())
                print(f"  {name} start {start} lr {lr} epoch {ep}: val KL {v:.5f} (start {v0:.5f}) [{time.time() - t0:.0f}s]",
                      flush=True)
            with torch.no_grad():
                for i, p in enumerate(params):
                    p.copy_(best[2][i])
            g_kl, g_ck = score(m, te_g, teach["g"])
            o_kl, _ = score(m, te_o, teach["o"])
            r = dict(lr=lr, best_epoch=best[1], val=best[0], val_start=float(v0), group=float(g_kl.mean()),
                     coarse=float(g_ck.mean()), other=float(o_kl.mean()))
            print(f"  -> {start} lr {lr}: {r}", flush=True)
            if best_overall is None or r["val"] < best_overall["val"]:
                best_overall = r
            del m, opt
            if DEV == "mps":
                torch.mps.empty_cache()
        res[start] = best_overall
        json.dump(res, open(os.path.join(OUT, f"ref_{name}_f{f}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
