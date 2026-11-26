"""N5 step 4: the recursive unit on the whole network (all 30 blocks of stages 3-4), keep fraction FRAC, no training.

For block b in order, O = original network, C = network with blocks < b already shrunk:
  S = top FRAC by value (std of GELU output x |write|) on O's calibration rows
  RE-READ  (variant full): kept rows of W1, b1 re-solved so that pre-activations on C's input u^C match O's (u^O W1_S + b1_S)
  RE-WRITE kept writes re-solved so the block's MLP output on C matches O's MLP output y^O (least squares + bias)
Variants: rewrite_seq (no RE-READ), full (RE-READ + RE-WRITE). The independent build (pair/allblocks.py) is the reference.
Calibration: 600 fit photos (one per class, random classes); rows: every 4th position in stage 3, all 49 in stage 4.
Test: the 500 test photos of diag.py; KL to Base and top-1.
Usage: python wholenet.py FRAC [variants...]
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import DEV, OUT, blocks, kl_rows, load, net, split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pair"))
from distill import D  # noqa: E402

torch.set_grad_enabled(False)


class Stop(Exception):
    pass


def main():
    frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    variants = sys.argv[2:] or ["rewrite_seq", "full"]
    ph = net.Photos("read")
    fit = split(ph.lab)
    cal = np.array(sorted([np.where((ph.lab == c) & fit)[0][8] for c in range(1000)]))
    cal = np.sort(np.random.default_rng(4).choice(cal, 600, replace=False))
    te = np.load(os.path.join(D, "photos_test.npy"))[:500]
    lab = torch.tensor(ph.lab[te])
    O, C = load("base"), load("base")
    BO, BC = blocks(O), blocks(C)
    orig = [[p.detach().clone() for p in (b.block[3].weight, b.block[3].bias, b.block[5].weight, b.block[5].bias)]
            for b in BC]
    ref = [O(ph.tensor(te[a:a + 50]).to(DEV)) for a in range(0, len(te), 50)]

    def evaluate():
        kl, c = [], 0
        for i, a in enumerate(range(0, len(te), 50)):
            y = C(ph.tensor(te[a:a + 50]).to(DEV))
            kl.append(kl_rows(ref[i], y).cpu())
            c += int((y.argmax(1).cpu() == lab[a:a + 50]).sum())
        return torch.cat(kl).numpy(), c / len(te)

    def rows(o, b):
        r = o.reshape(o.shape[0], -1, o.shape[-1])
        if b < 27:
            r = r[:, ::4]
        return r.reshape(-1, o.shape[-1]).float().cpu()

    def capture(net_blocks, model, b, what):
        H, out = {}, {k: [] for k in what}
        hs = [net_blocks[b].block[2].register_forward_hook(lambda mod, i, o: H.__setitem__("u", o)),
              net_blocks[b].block[4].register_forward_hook(lambda mod, i, o: H.__setitem__("g", o)),
              net_blocks[b].block[5].register_forward_hook(lambda mod, i, o: (H.__setitem__("y", o),
                                                                              (_ for _ in ()).throw(Stop())))]
        for a in range(0, len(cal), 50):
            try:
                model(ph.tensor(cal[a:a + 50]).to(DEV))
            except Stop:
                pass
            for k in what:
                out[k].append(rows(H[k], b))
        for h in hs:
            h.remove()
        return {k: torch.cat(v) for k, v in out.items()}

    res = {}
    for var in variants:
        for b, blk in enumerate(BC):
            for p, v in zip((blk.block[3].weight, blk.block[3].bias, blk.block[5].weight, blk.block[5].bias), orig[b]):
                p.copy_(v)
        t0 = time.time()
        shifts = []
        for b in range(len(BC)):
            blk = BC[b]
            ls = blk.layer_scale.detach().view(-1).cpu().double()
            W1, b1 = orig[b][0].cpu().double(), orig[b][1].cpu().double()
            Wb, bb = ls[:, None] * orig[b][2].cpu().double(), ls * orig[b][3].cpu().double()
            Wbf, bbf = Wb.float(), bb.float()
            o = capture(BO, O, b, ("u", "g"))
            c = capture(BC, C, b, ("u",))
            gO = o["g"]
            yO = gO @ Wbf.T + bbf  # O's MLP output (with layer scale), float32
            h = Wb.shape[1]
            kk = int(round(frac * h))
            S = torch.argsort(-(gO.std(0).double() * Wb.norm(dim=0)))[:kk]
            Uc = c["u"].double()  # (rows x d): d = 512 or 1024, small enough for float64
            W1n, b1n = W1.clone(), b1.clone()
            if var == "full":
                P = o["u"].double() @ W1[S].T + b1[S]
                U1 = torch.cat([Uc, torch.ones(len(Uc), 1, dtype=Uc.dtype)], 1)
                A = U1.T @ U1
                lam = 1e-4 * A.diagonal()[:-1].mean()
                Rg = torch.ones(len(A), dtype=A.dtype)
                Rg[-1] = 0
                old = torch.cat([W1[S].T, b1[S][None]], 0)
                sol = torch.linalg.solve(A + lam * torch.diag(Rg), U1.T @ P + lam * Rg[:, None] * old)
                W1n[S], b1n[S] = sol[:-1].T, sol[-1]
                shifts.append(float(((W1n[S] - W1[S]).norm(dim=1) / W1[S].norm(dim=1)).median()))
            gS = F.gelu(Uc @ W1n[S].T + b1n[S]).float()
            X1 = torch.cat([gS, torch.ones(len(gS), 1)], 1)
            A = (X1.T @ X1).double()
            A += 1e-4 * A.diagonal().mean() * torch.eye(len(A), dtype=A.dtype)
            beta = torch.linalg.solve(A, (X1.T @ yO).double())
            Wn = torch.zeros_like(Wb)
            Wn[:, S] = beta[:kk].T
            blk.block[3].weight.copy_(W1n.float().to(DEV))
            blk.block[3].bias.copy_(b1n.float().to(DEV))
            blk.block[5].weight.copy_((Wn / ls[:, None]).float().to(DEV))
            blk.block[5].bias.copy_((beta[kk] / ls).float().to(DEV))
            del o, c, gO, yO, Uc, gS, X1, A
            print(f"  {var} block {b} done [{time.time() - t0:.0f}s]", flush=True)
        kl, top1 = evaluate()
        res[var] = dict(kl=float(kl.mean()), top1=top1, reread_shift_median_per_block=shifts)
        np.save(os.path.join(OUT, f"wholenet_{var}_{frac}_rows.npy"), kl)
        print(f"{var} keep {frac}: KL {kl.mean():.4f}, top-1 {top1:.3f}", flush=True)
        json.dump(res, open(os.path.join(OUT, f"wholenet_{frac}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
