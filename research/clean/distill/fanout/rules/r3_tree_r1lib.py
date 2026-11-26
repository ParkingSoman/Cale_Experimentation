"""TREE r3: top R1 partner of each kept MLP unit (pure weights, rho = 1; r2_p1p3_merge._forest's votes).
R1 votes v_i[p] = (fc_w[i] * ln2.g) . w~_p over all upstream write columns p (centred over dims).
parent pi(i) = argmax_p v_i[p]. Partner of kept j = the CUT unit with the same parent and max cos(v_j, v_c);
if no cut unit shares j's parent, the cut unit with max cos(v_j, v_c) overall.
PHOTO USE: none (weights only)."""
import torch

from rules import r3_tree_lib as L

DEV = L.DEV


def make(T):
    @torch.no_grad()
    def r1(bi, kept, cutc, Gc):
        d = T["cls"].numel()
        cols = [torch.eye(d)]
        for b in range(bi + 1):
            cols.append(T["blocks"][b]["o_w"].float())
        for b in range(bi):
            cols.append(T["blocks"][b]["pr_w"].float())
        Wc = torch.cat(cols, 1)
        Wc = (Wc - Wc.mean(0, keepdim=True)).to(DEV)
        B = T["blocks"][bi]
        R = (B["fc_w"].float() * B["ln2"]["g"].float()[None]).to(DEV)
        V = R @ Wc
        pi = V.argmax(1).cpu()
        Vn = V / V.norm(dim=1, keepdim=True).clamp_min(1e-12)
        C = (Vn[kept.to(DEV)] @ Vn[cutc.to(DEV)].T).cpu()            # (k, ncut)
        del V, Vn, Wc
        same = pi[kept][:, None] == pi[cutc][None, :]
        cs = torch.where(same, C, torch.full_like(C, -9.0))
        ci = torch.where(same.any(1), cs.argmax(1), C.argmax(1))
        print(f"  r1 block {bi}: kept with a same-parent cut unit {float(same.any(1).float().mean()):.2f}", flush=True)
        return ci
    return r1
