"""CALC 3.6, Step 1 (plan id-like-you-to-fizzy-cherny, 2026-11-25): poles and feeders for the checklist reading.

Units: the 10 KD student neurons of CALC 3.5 (names from out/union_eye/*.png), their 30 teacher members, and a pool
of 40 random teacher neurons per teacher block (seed 11, members excluded) for the nulls.

Pass A, all 8,000 reading photos (fit positions 0-7, same ids as maxfire_*.npz):
  per photo and unit: max GELU output over the 197 tokens and its token; min c_fc pre-activation and its token.
Pass B, per unit, its top 40 FIT crops (max token) and its 40 most negative FIT crops (min token):
  feeders = the direct contribution of every upstream unit to the unit's pre-activation at that token:
    contribution of a stream component c = c . r' / sigma, with r' = ln_2.weight * W_fc[j] minus its mean,
    sigma = the token's ln_2 scale. Components: the embedding (ln_pre output), each attention head's write, each
    block's attention bias, each MLP neuron's write (GELU out x W_proj column), each block's MLP bias.
    Constant = ln_2.bias . W_fc[j] + b_fc[j]. Sum + constant = pre-activation (checked).
  cross pre-activations: at each crop's photo and token, the pre-activation of every unit (teacher units from the
    teacher, student units from the KD student).
Photos for reading only. FIT = positions 0-3, SCORE = 4-7. Old files are not touched.
Output: out/poles[_smoke].npz
Usage: PYTHONPATH=distill/clipkd/pylib curves/.venv/bin/python -I distill/clipkd/read_poles.py [--smoke]
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_clip import CLIP_MEAN, CLIP_STD, DEV, OUT, net  # noqa: E402
from read_stream import tower  # noqa: E402

NT = 197
Q = 40  # crops per pole


def named_units():
    """(model, block0, neuron, role, group) from the 3.5 eye-sheet names. role: student / single / m2 / m3."""
    units = []
    for g, f in enumerate(sorted(glob.glob(os.path.join(OUT, "union_eye", "*.png")))):
        m = re.match(r"b(\d+)_kd(\d+)_t(.+)\.png", os.path.basename(f))
        b = int(m.group(1)) - 1
        units.append(("kd", b, int(m.group(2)), "student", g))
        for role, tok in zip(("single", "m2", "m3"), m.group(3).split("-b")):
            tok = tok.lstrip("b")
            if tok.startswith("-1:"):
                units.append(("teacher", b - 1, int(tok[3:]), role, g))
            else:
                units.append(("teacher", b, int(tok), role, g))
    return units


def random_pool(units, per_block=40, seed=11):
    rng = np.random.default_rng(seed)
    out = []
    for b in sorted({u[1] for u in units if u[0] == "teacher"}):
        taken = {u[2] for u in units if u[0] == "teacher" and u[1] == b}
        cand = np.array([i for i in range(3072) if i not in taken])
        for j in rng.choice(cand, per_block, replace=False):
            out.append(("teacher", b, int(j), "random", -1))
    return out


class Grab:
    """Hooks that keep, per block, only the requested (photo-in-batch, token) rows, or the per-photo poles."""

    def __init__(self, v, mode, want):
        self.v, self.mode, self.want = v, mode, want  # want: {block: neuron index tensor} for pass A
        self.req = None  # pass B: (bi, ti) long tensors
        self.store = {}
        self.hs = []
        bl = v.transformer.resblocks
        for b, blk in enumerate(bl):
            assert isinstance(blk.ls_1, torch.nn.Identity) and isinstance(blk.ls_2, torch.nn.Identity)
            assert not hasattr(blk, "ln_attn"), "unexpected block type"
            self.hs.append(blk.mlp.c_fc.register_forward_hook(self._mk("pre", b)))
            self.hs.append(blk.mlp.gelu.register_forward_hook(self._mk("gelu", b)))
            if mode == "B":
                self.hs.append(blk.attn.out_proj.register_forward_pre_hook(self._mk_pre("z", b)))
                self.hs.append(blk.ln_2.register_forward_pre_hook(self._mk_pre("mid", b)))
        if mode == "B":
            self.hs.append(v.ln_pre.register_forward_hook(self._mk("emb", -1)))

    def _bf(self, o, n):
        if o.shape[0] == NT and o.shape[1] == n:
            o = o.transpose(0, 1)
        assert o.shape[1] == NT, o.shape
        return o

    def _keep(self, key, b, o):
        o = self._bf(o, self.n)
        if self.mode == "A":
            if b not in self.want:
                return
            o = o[:, :, self.want[b]].float()
            if key == "gelu":
                mv, mt = o.max(1)
                self.store[("gmax", b)], self.store[("gtok", b)] = mv.cpu(), mt.cpu()
            else:
                mv, mt = o.min(1)
                self.store[("pmin", b)], self.store[("ptok", b)] = mv.cpu(), mt.cpu()
        else:
            self.store[(key, b)] = o[self.req[0], self.req[1]].float().cpu()

    def _mk(self, key, b):
        return lambda m, i, o: self._keep(key, b, o)

    def _mk_pre(self, key, b):
        return lambda m, i: self._keep(key, b, i[0])

    def run(self, x, req=None):
        self.n, self.req, self.store = x.shape[0], req, {}
        with torch.no_grad():
            self.v(x)
        return self.store

    def close(self):
        for h in self.hs:
            h.remove()


def weights(v):
    f = lambda t: t.detach().float().cpu().numpy()
    W = []
    for blk in v.transformer.resblocks:
        W.append(dict(Wfc=f(blk.mlp.c_fc.weight), bfc=f(blk.mlp.c_fc.bias), Wp=f(blk.mlp.c_proj.weight),
                      bp=f(blk.mlp.c_proj.bias), Wo=f(blk.attn.out_proj.weight), bo=f(blk.attn.out_proj.bias),
                      g2=f(blk.ln_2.weight), be2=f(blk.ln_2.bias), eps=blk.ln_2.eps, heads=blk.attn.num_heads))
    return W


def layout(W):
    """Feeder index layout: [emb] then per block [heads..., attn bias, mlp neurons..., mlp bias]."""
    off, names = [1], ["emb"]
    for b, w in enumerate(W):
        H, M = w["heads"], w["Wp"].shape[1]
        names += [f"b{b}:head{h}" for h in range(H)] + [f"b{b}:attn_bias"] + [f"b{b}:mlp{i}" for i in range(M)] + [f"b{b}:mlp_bias"]
        off.append(off[-1] + H + 1 + M + 1)
    return off, names


def check_mapping(v, ph, ids):
    """Token t <-> patch row (t-1)//14, column (t-1)%14: change one patch, only that token's embedding moves."""
    out = []
    h = v.ln_pre.register_forward_hook(lambda m, i, o: out.append(o))
    raw = ph.raw(ids[:1]).copy()
    for rr, cc in ((0, 0), (3, 11), (13, 5)):
        r2 = raw.copy(); r2[0, rr * 16:(rr + 1) * 16, cc * 16:(cc + 1) * 16] = 255 - r2[0, rr * 16:(rr + 1) * 16, cc * 16:(cc + 1) * 16]
        out.clear()
        for r_ in (raw, r2):
            x = torch.from_numpy(r_).permute(0, 3, 1, 2).float() / 255.0
            with torch.no_grad():
                v(((x - CLIP_MEAN) / CLIP_STD).to(DEV))
        d = (out[0] - out[1]).abs().float()
        if d.shape[0] == NT:
            d = d.transpose(0, 1)
        moved = torch.where(d[0].sum(-1) > 1e-4)[0].cpu().tolist()
        assert moved == [1 + rr * 14 + cc], (rr, cc, moved)
    h.remove()
    print("check mapping: PASS", flush=True)


def tensor(ph, ids):
    x = torch.from_numpy(ph.raw(ids).copy()).permute(0, 3, 1, 2).float() / 255.0
    return ((x - CLIP_MEAN) / CLIP_STD).to(DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bs", type=int, default=40)
    a = ap.parse_args()
    t0 = time.time()
    tag = "_smoke" if a.smoke else ""
    mf = {n: np.load(os.path.join(OUT, f"maxfire_{n}.npz")) for n in ("teacher", "kd")}
    ids, pos = mf["kd"]["photo_ids"], mf["kd"]["pos"]
    assert (ids == mf["teacher"]["photo_ids"]).all()
    if a.smoke:
        keep = np.arange(25 * 8)
        ids, pos = ids[keep], pos[keep]
    else:
        keep = np.arange(len(ids))
    fit = pos < 4
    units = named_units()
    assert len(units) == 40, len(units)
    units += random_pool(units)
    ph = net.Photos("read")
    U = np.array([(u[0] == "kd", u[1], u[2]) for u in units])
    res = dict(unit_model=np.array([u[0] for u in units]), unit_block=U[:, 1], unit_neuron=U[:, 2],
               unit_role=np.array([u[3] for u in units]), unit_group=np.array([u[4] for u in units]),
               photo_ids=ids, pos=pos)

    # ---------- Pass A ----------
    N, NU = len(ids), len(units)
    gmax = np.zeros((N, NU), np.float32); gtok = np.zeros((N, NU), np.int16)
    pmin = np.zeros((N, NU), np.float32); ptok = np.zeros((N, NU), np.int16)
    for mname in ("teacher", "kd"):
        v = tower(mname)
        check_mapping(v, ph, ids)
        sel = [k for k, u in enumerate(units) if u[0] == mname]
        want = {}
        for k in sel:
            want.setdefault(units[k][1], []).append(k)
        wt = {b: torch.tensor([units[k][2] for k in ks], device=DEV) for b, ks in want.items()}
        G = Grab(v, "A", wt)
        for i in range(0, N, a.bs):
            s = G.run(tensor(ph, ids[i:i + a.bs]))
            for b, ks in want.items():
                gmax[i:i + a.bs, ks] = s[("gmax", b)].numpy(); gtok[i:i + a.bs, ks] = s[("gtok", b)].numpy()
                pmin[i:i + a.bs, ks] = s[("pmin", b)].numpy(); ptok[i:i + a.bs, ks] = s[("ptok", b)].numpy()
        G.close()
        # check: the max equals the saved maxfire file
        ref = np.stack([mf[mname]["fire"][keep, units[k][1], units[k][2]].astype(np.float32) for k in sel], 1)
        err = np.abs(ref - gmax[:, sel]).max() / (np.abs(ref).max() + 1e-6)
        print(f"A {mname}: {len(sel)} units, max rel diff vs maxfire {err:.4f}, {time.time() - t0:.0f}s", flush=True)
        assert err < 0.01, err
        res[f"W_{mname}"] = None
        del v
        if DEV == "mps":
            torch.mps.empty_cache()
    res.update(gmax=gmax, gtok=gtok, pmin=pmin, ptok=ptok)
    print("class-token share of max tokens: named", float((gtok[:, :40] == 0).mean()), "pool", float((gtok[:, 40:] == 0).mean()))

    # ---------- crops ----------
    fi = np.where(fit)[0]
    crops = []  # (unit, pole, photo row, token)
    for k in range(NU):
        for r in fi[np.argsort(-gmax[fi, k])[:Q]]:
            crops.append((k, 0, r, gtok[r, k]))
        for r in fi[np.argsort(pmin[fi, k])[:Q]]:
            crops.append((k, 1, r, ptok[r, k]))
    crops = np.array(crops, np.int64)
    res["crops"] = crops
    rows = np.unique(crops[:, 2])

    # ---------- Pass B ----------
    cross = np.zeros((len(crops), NU), np.float32)
    feed_sum, recon_err = {}, np.zeros(len(crops), np.float32)
    for mname in ("teacher", "kd"):
        v = tower(mname)
        W = weights(v)
        off, names = layout(W)
        res[f"feeder_names_{mname}"] = np.array(names)
        sel = [k for k, u in enumerate(units) if u[0] == mname]
        # per owner unit: read r' and its back-projections
        R = {}
        for k in sel:
            b, j = units[k][1], units[k][2]
            w = W[b]
            r = w["g2"] * w["Wfc"][j]
            r = r - r.mean()
            R[k] = dict(r=r, const=float(w["be2"] @ w["Wfc"][j] + w["bfc"][j]),
                        o=[W[c]["Wo"].T @ r for c in range(b + 1)],
                        p=[W[c]["Wp"].T @ r for c in range(b)],
                        bo=[float(W[c]["bo"] @ r) for c in range(b + 1)],
                        bp=[float(W[c]["bp"] @ r) for c in range(b)])
        acc = {(k, p): np.zeros(off[-1], np.float64) for k in sel for p in (0, 1)}
        G = Grab(v, "B", None)
        for i in range(0, len(rows), a.bs):
            br = rows[i:i + a.bs]
            ci = np.where(np.isin(crops[:, 2], br))[0]
            bi = torch.tensor(np.searchsorted(br, crops[ci, 2]), device=DEV)
            ti = torch.tensor(crops[ci, 3], device=DEV)
            s = G.run(tensor(ph, ids[br]), (bi, ti))
            for k in sel:  # cross pre-activations
                cross[ci, k] = s[("pre", units[k][1])][:, units[k][2]].numpy()
            for n, c in enumerate(ci):  # feeders, only for crops owned by this model's units
                k, p = crops[c, 0], crops[c, 1]
                if k not in R:
                    continue
                b = units[k][1]
                Rk = R[k]
                mid = s[("mid", b)][n].numpy()
                sig = np.sqrt(mid.var() + W[b]["eps"])
                f = np.zeros(off[-1], np.float64)
                f[0] = s[("emb", -1)][n].numpy() @ Rk["r"]
                for c2 in range(b + 1):
                    H = W[c2]["heads"]; z = s[("z", c2)][n].numpy(); dh = len(z) // H
                    o0 = off[c2]
                    f[o0:o0 + H] = (z * Rk["o"][c2]).reshape(H, dh).sum(1)
                    f[o0 + H] = Rk["bo"][c2]
                    if c2 < b:
                        g = s[("gelu", c2)][n].numpy(); M = len(g)
                        f[o0 + H + 1:o0 + H + 1 + M] = g * Rk["p"][c2]
                        f[o0 + H + 1 + M] = Rk["bp"][c2]
                f /= sig
                recon_err[c] = abs(f.sum() + Rk["const"] - cross[c, k]) / (abs(cross[c, k]) + 1e-3)
                acc[(k, p)] += f
            if i % (10 * a.bs) == 0:
                print(f"B {mname} {i}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
        G.close()
        for k in sel:
            for p in (0, 1):
                n = ((crops[:, 0] == k) & (crops[:, 1] == p)).sum()
                feed_sum[(k, p)] = (acc[(k, p)] / n).astype(np.float32)
        res[f"feed_{mname}_units"] = np.array(sel)
        res[f"feed_{mname}"] = np.stack([[feed_sum[(k, 0)], feed_sum[(k, 1)]] for k in sel])
        res[f"feed_{mname}_const"] = np.array([R[k]["const"] for k in sel], np.float32)
        del v
        if DEV == "mps":
            torch.mps.empty_cache()
    # check: the crop's own pre-activation equals the saved pole value
    own = cross[np.arange(len(crops)), crops[:, 0]]
    pole_val = np.where(crops[:, 1] == 1, pmin[crops[:, 2], crops[:, 0]], np.nan)
    m = crops[:, 1] == 1
    print("check min pole: max abs diff", float(np.abs(own[m] - pole_val[m]).max()))
    print("check feeders: median rel recon err", float(np.median(recon_err)), "p99", float(np.percentile(recon_err, 99)))
    res["cross"], res["recon_err"] = cross, recon_err
    res = {k: v for k, v in res.items() if v is not None}
    np.savez(os.path.join(OUT, f"poles{tag}.npz"), **res)
    print("saved", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
