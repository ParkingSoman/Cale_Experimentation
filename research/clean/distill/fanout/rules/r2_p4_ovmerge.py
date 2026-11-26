"""P4 round 2: CLIP heads-only, a better WRITE form for heads: OV MERGE.

Round-1 heads fold predicted a cut head's output from kept heads' outputs at the same token: about no gain (calibration).
A head's message is O_h V_h y averaged by its OWN attention pattern A_h. If a kept head j has a pattern close to A_c,
the cut head's message is about A_j (O_c V_c) y, so it can ride on head j's pattern:
  M_j = O_j V_j + sum_{c -> j} O_c V_c  (768 x 768, rank up to 64 (1 + #c)),
  O'_j V'_j = best rank-64 approximation of M_j in the metric of the head input y = ln1(x): C = E[y y^T] on X.
  SVD of M_j C^(1/2) = U S Vt; O'_j = U_64 S_64^(1/2); V'_j = S_64^(1/2) Vt_64 C^(-1/2) (C ridge 1e-6 tr/768).
  Value biases move into o_b (exact: attention rows sum to 1).
Partner of cut head c = the kept head with the most similar attention pattern on X (mean cosine of A over images).
null="rand": partner = a random kept head (same form). Kept heads = the floor's ranking on X.
Derived from: min E|| M_j y - O'_j V'_j y ||^2 over rank-64 maps, if A_c = A_j (Eckart-Young in the C metric).
Object (picked by): v_w, v_b, o_w, o_b of every kept head that receives a partner (floor ranking).
PHOTO USE: none. Weights + generic input X.
"""
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
FAN = os.path.dirname(HERE)
sys.path.insert(0, FAN)
import vitgen  # noqa: E402
from rules import noisecut, p4write as P  # noqa: E402

DEV = noisecut.DEV
HD = vitgen.HEAD_DIM


@torch.no_grad()
def build(T, f, X=None, null=None, tag="ovmerge"):
    X = P._x_vit(X)
    sc = P._vit_scores(T, X)
    D, H, M = vitgen.sizes(1.0, T)
    _, h, _ = vitgen.sizes(f, T)
    hk = [noisecut.top(s, h) for s in sc["head"]]
    W = vitgen.cut(T, torch.arange(D), hk, [torch.arange(M)] * len(T["blocks"]))
    Td = vitgen.to(T, DEV)
    nX = len(X)
    sls = P._slices(nX)
    t = torch.cat([vitgen.embed(Td, X[sl].to(DEV).float()) for sl in sls])
    g = torch.Generator().manual_seed(0)
    nmerge = 0
    for bi, BT in enumerate(Td["blocks"]):
        kept = hk[bi].tolist()
        cut = [c for c in range(H) if c not in kept]
        C = torch.zeros(D, D, dtype=torch.float64)
        sim = torch.zeros(H, H, dtype=torch.float64)
        N = 0
        for sl in sls:
            y = vitgen.norm(t[sl], BT["ln1"])
            Nb, L, _ = y.shape
            q = F.linear(y, BT["q_w"], BT["q_b"]).view(Nb, L, H, HD).transpose(1, 2)
            k = F.linear(y, BT["k_w"], BT["k_b"]).view(Nb, L, H, HD).transpose(1, 2)
            A = torch.softmax(q @ k.transpose(-1, -2) / HD ** .5, -1).flatten(2)        # (Nb, H, L*L)
            A = F.normalize(A, dim=-1)
            sim += torch.einsum("nhx,ngx->hg", A, A).cpu().double()
            yf = y.flatten(0, 1)
            C += (yf.T @ yf).cpu().double()
            N += yf.shape[0]
        C /= N
        ev, evec = torch.linalg.eigh(C + 1e-6 * C.trace() / D * torch.eye(D, dtype=torch.float64))
        Ch, Cih = (evec * ev.sqrt()) @ evec.T, (evec / ev.sqrt()) @ evec.T
        part = {}
        for c in cut:
            if null == "rand":
                j = kept[int(torch.randint(len(kept), (1,), generator=g))]
            else:
                j = max(kept, key=lambda j_: float(sim[c, j_]))
            part.setdefault(j, []).append(c)
        B = W["blocks"][bi]
        BTc = T["blocks"][bi]
        ob = BTc["o_b"].double().clone()
        for pos, j in enumerate(kept):          # every value bias goes into o_b (exact)
            rj = slice(j * HD, (j + 1) * HD)
            ob += BTc["o_w"][:, rj].double() @ BTc["v_b"][rj].double()
        for j, cs in part.items():
            pos = kept.index(j)
            Mj = sum(BTc["o_w"][:, hh * HD:(hh + 1) * HD].double() @ BTc["v_w"][hh * HD:(hh + 1) * HD].double()
                     for hh in [j] + cs)
            for c in cs:
                rc = slice(c * HD, (c + 1) * HD)
                ob += BTc["o_w"][:, rc].double() @ BTc["v_b"][rc].double()
            U, S, Vt = torch.linalg.svd(Mj @ Ch)
            O2 = U[:, :HD] * S[:HD].sqrt()
            V2 = (S[:HD].sqrt()[:, None] * Vt[:HD]) @ Cih
            rs = slice(pos * HD, (pos + 1) * HD)
            B["o_w"][:, rs] = O2.float()
            B["v_w"][rs] = V2.float()
            nmerge += len(cs)
        B["v_b"] = torch.zeros_like(B["v_b"])
        B["o_b"] = ob.float()
        for sl in sls:
            t[sl] = vitgen.block(t[sl], BT)
    print(f"[r2_p4 {tag} f={f:g}] merged {nmerge} cut heads", flush=True)
    del Td, t
    return W


def vit(T, f, X=None):
    return build(T, f, X)
