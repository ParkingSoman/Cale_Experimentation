import sys, os, torch
FAN = "/Users/shonusengupta/Cale_Experimentation/research/clean/distill/fanout"
sys.path.insert(0, FAN); sys.path.insert(0, os.path.dirname(FAN))
from rules import r3_joint_clip as J, p4write as P4, l1_stream as L
import board
T = board.clip_teacher()
X = P4._x_vit(torch.load(os.path.join(FAN, "cache", "input_pink_n256_s1.pt"), weights_only=True).float())
D = 768
for f in (.85, .5, .25):
    _, hk, mk = J.keep_units(T, f, X, "act", "w", 0)
    r99w, r99r = [], []
    for B, h, m in zip(T["blocks"], hk, mk):
        Ww = torch.cat([B["o_w"][:, L._rows(h)], B["pr_w"][:, m]], 1).double()
        Wr = torch.cat([B["q_w"][L._rows(h)], B["k_w"][L._rows(h)], B["v_w"][L._rows(h)], B["fc_w"][m]]).double()
        for W, out in ((Ww, r99w), (Wr.T, r99r)):
            sv = torch.linalg.svdvals(W) ** 2
            c = sv.cumsum(0) / sv.sum()
            out.append(int((c < .99).sum()) + 1)
    print(f"f={f}: per-block dims holding 99% of kept WRITE weight energy {r99w}; READ {r99r}")
