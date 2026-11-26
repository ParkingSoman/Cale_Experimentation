"""LABELLED READING CHECK (photos, off the board). P1-P3 r2: the gate's target rate on pink vs on held photos.
Object (picked by): CLIP MLP neurons, kept set = floor order on pink; groups = R1 (weighted by noise RMS) forest on pink.
For each f: mean over blocks of (kept unit's own rate z>0) and (group union rate) on 8,192 tokens of pink X and of
64 held photos. Photo use: 64 held photos, for this reading only. Nothing here enters a calculation."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FAN = os.path.dirname(HERE)
sys.path.insert(0, FAN)
sys.path.insert(0, os.path.dirname(FAN))
sys.path.insert(0, os.path.join(os.path.dirname(FAN), "clipkd"))
import board  # noqa: E402
import eval_fast  # noqa: E402
import vitgen  # noqa: E402
from rules import p4write, r2_p1p3_merge as M  # noqa: E402

T = board.clip_teacher()
Xp = torch.load(os.path.join(FAN, "cache", "input_pink_n256_s1.pt"), weights_only=True).float()
ref = eval_fast.teacher_ref("held")
Xh = eval_fast.photos(ref["ids"][:64]).float()
sp, sh = M._teacher_stats(T, Xp), M._teacher_stats(T, Xh)
sc = p4write._vit_scores(T, Xp)
for f in (0.95, 0.85, 0.5):
    _, _, mn = vitgen.sizes(f, T)
    rows = []
    for bi, B in enumerate(T["blocks"]):
        kept = M.noisecut.top(sc["mlp"][bi], mn)
        m = B["fc_w"].shape[0]
        km = torch.zeros(m, dtype=torch.bool)
        km[kept] = True
        key = ("forest", "r1rms", bi, id(Xp))
        if key not in M._C:
            M._C[key] = M._forest(T, sp, bi, sc["mlp"][bi], "r1rms")
        own = M._owner(M._C[key][0], km)
        col = torch.full((m,), -1, dtype=torch.long)
        col[kept] = torch.arange(len(kept))
        A = torch.zeros(m, len(kept))
        A[torch.arange(m), col[own]] = 1.0
        r = []
        for st in (sp, sh):
            Z = st["y"][bi].float() @ B["fc_w"].float().T + B["fc_b"].float()
            r += [float((Z[:, kept] > 0).float().mean()), float(((((Z > 0).float() @ A) > 0).float()).mean())]
        rows.append(r)
    R = torch.tensor(rows)
    print(f"f={f}: own rate pink {R[:,0].mean():.3f} photo {R[:,2].mean():.3f} | union rate pink {R[:,1].mean():.3f} "
          f"photo {R[:,3].mean():.3f} | per block union pink/photo " + " ".join(f"{a:.2f}" for a in (R[:, 1] / R[:, 3]).tolist()),
          flush=True)
