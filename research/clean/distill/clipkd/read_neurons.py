"""Step 3 of plan "read the CLIP-KD pair" (2026-11-25): does each student MLP neuron match one teacher MLP neuron,
or a small group of them (the essay's coarser neuron covering a family)?

Photos for reading only: the same 2,000 reading photos as read_stream.py (fit positions 0 and 1 of each class).
  FIT half picks the match. SCORE half reports it.
Firing = MLP GELU output (teacher 3,072 per block, students 768), every one of 197 tokens, same block depth.
Per block b and student s (kd, baseline), per student neuron j:
  single  = correlation with the teacher neuron that correlates best on FIT, measured on SCORE.
  group8  = R^2 on SCORE of j from its top-8 teacher neurons (by FIT correlation) + constant, least squares on FIT.
            Only blocks GB (token subsample: class token + 8 fixed patch positions per photo).
Control: the baseline student (no teacher), read the same way. Picking the best match passes against random by
construction, so the baseline is the null, not random groups.
Output: out/read_neurons.json (table), out/read_neurons_corr.npz (SCORE-half correlation matrices, fp16, all blocks).
Usage: PYTHONPATH=distill/clipkd/pylib curves/.venv/bin/python -I distill/clipkd/read_neurons.py [--smoke]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_clip import CLIP_MEAN, CLIP_STD, DEV, OUT, net  # noqa: E402
from read_stream import STUDENTS, photo_halves, tower  # noqa: E402

NB = 12
GB = (3, 6, 9, 12)  # 1-indexed blocks for the group test
NSUB = 8
M = 8


def firings(v, x):
    out = []
    hs = [b.mlp.gelu.register_forward_hook(lambda m, i, o: out.append(o)) for b in v.transformer.resblocks]
    with torch.no_grad():
        v(x)
    for h in hs:
        h.remove()
    res = []
    for o in out:
        if o.shape[0] == 197 and o.shape[1] == x.shape[0]:
            o = o.transpose(0, 1)
        assert o.shape[1] == 197, o.shape
        res.append(o.float())  # B x 197 x D
    return res


class Corr:
    """Sums for the correlation of every teacher neuron with every student neuron (on the device, float32)."""
    def __init__(self, dt, ds):
        z = lambda *s: torch.zeros(*s, device=DEV)
        self.st, self.stt, self.ss, self.sss, self.sts, self.n = z(dt), z(dt), z(ds), z(ds), z(dt, ds), 0

    def add(self, T, S):
        mt, ms = T.mean(0), S.mean(0)  # shift by the batch mean for float32 precision, then merge exactly
        Tc, Sc = T - mt, S - ms
        n2 = len(T)
        if self.n == 0:
            self.mt, self.ms = mt, ms
        dmt, dms = mt - self.mt, ms - self.ms
        # sums around the running reference means (self.mt, self.ms), kept fixed after the first batch
        self.st += (Tc + dmt).sum(0); self.ss += (Sc + dms).sum(0)
        self.stt += (Tc ** 2).sum(0) + 2 * dmt * Tc.sum(0) + n2 * dmt ** 2
        self.sss += (Sc ** 2).sum(0) + 2 * dms * Sc.sum(0) + n2 * dms ** 2
        self.sts += Tc.T @ Sc + torch.outer(dmt, Sc.sum(0)) + torch.outer(Tc.sum(0), dms) + n2 * torch.outer(dmt, dms)
        self.n += n2

    def corr(self):
        n = self.n
        mt, ms = self.st / n, self.ss / n
        vt, vs = self.stt / n - mt ** 2, self.sss / n - ms ** 2
        cov = self.sts / n - torch.outer(mt, ms)
        return (cov / torch.sqrt(torch.outer(vt.clamp_min(1e-12), vs.clamp_min(1e-12)))).cpu()


def group_r2(Tf, Sf, Ts, Ss, idx, lam=1e-3):
    """Per student neuron j: least squares of S[:, j] on T[:, idx[j]] + 1, fit on f, R^2 on s."""
    J = Sf.shape[1]
    out = np.zeros(J)
    for j in range(J):
        Xf = torch.cat([Tf[:, idx[j]], torch.ones(len(Tf), 1)], 1).double()
        Xs = torch.cat([Ts[:, idx[j]], torch.ones(len(Ts), 1)], 1).double()
        G = Xf.T @ Xf
        w = torch.linalg.solve(G + lam * G.diagonal().mean() * torch.eye(len(G), dtype=torch.float64), Xf.T @ Sf[:, j].double())
        e = Ss[:, j].double() - Xs @ w
        out[j] = float(1 - e.var() / Ss[:, j].double().var())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bs", type=int, default=25)
    a = ap.parse_args()
    t0 = time.time()
    ph = net.Photos("read")
    n = 50 if a.smoke else 1000
    halves = dict(zip(("fit", "score"), photo_halves(ph, n)))
    T = tower("teacher")
    S = {s: tower(s) for s in STUDENTS}
    g = torch.Generator().manual_seed(0)
    sub = torch.cat([torch.zeros(1, dtype=torch.long), 1 + torch.randperm(196, generator=g)[:NSUB]])
    acc = {(h, b, s): Corr(3072, 768) for h in halves for b in range(NB) for s in STUDENTS}
    keep = {(h, b, who): [] for h in halves for b in GB for who in ("teacher",) + STUDENTS}
    for h, ids in halves.items():
        for i in range(0, len(ids), a.bs):
            raw = ph.raw(ids[i:i + a.bs])
            x = torch.from_numpy(raw.copy()).permute(0, 3, 1, 2).float() / 255.0
            x = ((x - CLIP_MEAN) / CLIP_STD).to(DEV)
            FT = firings(T, x)
            for b in GB:
                keep[(h, b, "teacher")].append(FT[b - 1][:, sub].reshape(-1, 3072).half().cpu())
            for s in STUDENTS:
                FS = firings(S[s], x)
                for b in range(NB):
                    acc[(h, b, s)].add(FT[b].reshape(-1, 3072), FS[b].reshape(-1, 768))
                for b in GB:
                    keep[(h, b, s)].append(FS[b - 1][:, sub].reshape(-1, 768).half().cpu())
            print(f"{h} {i + x.shape[0]}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
    res = {"photos": {h: int(len(ids)) for h, ids in halves.items()}, "group_m": M, "group_blocks": list(GB), "blocks": {}}
    corr_save = {}
    for b in range(NB):
        row = {}
        for s in STUDENTS:
            cf, cs = acc[("fit", b, s)].corr(), acc[("score", b, s)].corr()  # 3072 x 768
            pick = cf.argmax(0)
            single = cs[pick, torch.arange(768)]
            row[s] = {"single_median": float(single.median()), "single_q90": float(single.quantile(.9)),
                      "share_single_ge_.7": float((single >= .7).float().mean()),
                      "share_single_ge_.5": float((single >= .5).float().mean()),
                      "teachers_used_by_best_match": int(len(set(pick.tolist())))}
            corr_save[f"b{b + 1}_{s}"] = cs.half().numpy()
            if (b + 1) in GB:
                top = cf.abs().topk(M, dim=0).indices.T  # 768 x M
                Tf, Ts = torch.cat(keep[("fit", b + 1, "teacher")]).float(), torch.cat(keep[("score", b + 1, "teacher")]).float()
                Sf, Ss = torch.cat(keep[("fit", b + 1, s)]).float(), torch.cat(keep[("score", b + 1, s)]).float()
                r2g = group_r2(Tf, Sf, Ts, Ss, top)
                r21 = group_r2(Tf, Sf, Ts, Ss, top[:, :1])
                row[s].update({"group8_R2_median": float(np.median(r2g)), "single_R2_median_sub": float(np.median(r21)),
                               "group8_gain_median": float(np.median(r2g - r21))})
        res["blocks"][f"b{b + 1}"] = row
        print(f"b{b + 1}", json.dumps(row), flush=True)
    tag = "_smoke" if a.smoke else ""
    json.dump(res, open(os.path.join(OUT, f"read_neurons{tag}.json"), "w"), indent=1)
    np.savez(os.path.join(OUT, f"read_neurons_corr{tag}.npz"), **corr_save)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
