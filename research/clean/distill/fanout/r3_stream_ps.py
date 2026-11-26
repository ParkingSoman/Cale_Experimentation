"""STREAM round 3 (FANOUT): CLIP stream-only, per-site reader-weighted bases + the where-it-dies diagnostic.

Object (picked by): the stream basis per site j (25 sites: ln1 x12, ln2 x12, post), task text round 3 (P6 for_round3 2).
Modes (--mode):
  glob     KNOWN ANSWER: every site gets the ONE global rG basis (r2_p6_core 'rG'); skip maps = I; must equal r2_p6_rG_rho.
  ps       per-site rG: Q_j = top-k eigvecs of G_j^1/2 C_j G_j^1/2, G_j = sum over ALL downstream reader sites l >= j of
           G_l / tr G_l (the content at site j is read at j and, through the skip, at every later site),
           C_j = M_j / tr M_j + ridge * I / D (ridge: the post site has rank <= n class tokens).
  psG      per-site G_j, pooled C (as global rG): isolates the per-site reader part.
  derange  NULL: the ps bases assigned to a derangement of the sites (same form, same bases, wrong place).
  rand     NULL: random per-site bases (r2_p6_persite 'rand').
  where    WHERE-IT-DIES: teacher at full width; from block b on the stream is projected onto the global rG Q
           (writes projected too), every norm from block b on is rms with divisor 768 * rho_l (as r2_p6_rG_rho);
           --b list. b = 12 is the teacher except ln_post (rms, D * rho_post).
Build and skip maps: r2_p6_persite.build form (copied: it takes a list of bases here). Own forward (skip maps).
Input X only (--input). Photos: scoring only. Rows: fanout/scratch.jsonl (quick) or fanout/r3_stream_rows.jsonl (full,
with per-photo KL and agree in out/).
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import board  # noqa: E402
from rules import r2_p6_core as core, r2_p6_persite as ps, _p6diag_clip  # noqa: E402
import vitgen  # noqa: E402

dt = torch.float64


def _sqrt(A):
    e, V = torch.linalg.eigh(A)
    return (V * e.clamp_min(0).sqrt()) @ V.T


def _top(A, k):
    e, V = torch.linalg.eigh(A)
    return core._orth_centred(V[:, e.argsort(descending=True)[:k]])


def bases(T, X, k, mode, ridge=1e-3, seed=0):
    D, nb = T["cls"].numel(), len(T["blocks"])
    ns = 2 * nb + 1
    if mode == "glob":
        Q, _ = core.basis(T, X, k, "rG")
        return [Q] * ns
    if mode == "rand":
        return [ps.site_basis(torch.eye(D, dtype=dt), k, "rand", j, seed) for j in range(ns)]
    m = core.site_moments(T, X)
    M = m["M"]
    Gn = [G / G.trace() for G in core.readers(T)]
    Cn = [M[j] / M[j].trace() for j in range(ns)]
    Cpool = sum(Cn)
    I = torch.eye(D, dtype=dt)
    Qs = []
    for j in range(ns):
        Gj = sum(Gn[j:])
        Gh = _sqrt(Gj)
        C = Cpool if mode == "psG" else Cn[j] + ridge * I / D
        Qs.append(_top(Gh @ C @ Gh, k))
    if mode == "derange":
        g = torch.Generator().manual_seed(seed)
        while True:
            p = torch.randperm(ns, generator=g)
            if not (p == torch.arange(ns)).any():
                break
        Qs = [Qs[int(i)] for i in p]
    return Qs


@torch.no_grad()
def build_ps(T, f, X, Qs):
    """r2_p6_persite.build with the bases given (copied, unchanged otherwise)."""
    D, nb = T["cls"].numel(), len(T["blocks"])
    k = Qs[0].shape[1]
    m = core.site_moments(T, X)
    J = torch.eye(D, dtype=dt) - 1.0 / D
    g = lambda v: v.to(dt)                                 # noqa: E731
    c32 = lambda v: v.float().contiguous()                 # noqa: E731
    Kp = Qs[0].T @ J @ torch.diag(g(T["ln_pre"]["g"])) @ J
    U, S, Vh = torch.linalg.svd(Kp, full_matrices=False)
    V = Vh.T
    Rs = [Qs[0] @ U] + Qs[1:]
    Ps = [U.T @ Qs[0].T @ J] + [Q.T @ J for Q in Qs[1:]]
    rho = [float(torch.trace(Q.T @ m["M"][j] @ Q) / torch.trace(m["M"][j])) for j, Q in enumerate(Qs)]
    st = _p6diag_clip._stats(T, X)
    rho_e = float(torch.trace(V.T @ st["Ct"] @ V) / torch.trace(st["Ct"]))
    ones, zeros = torch.ones(k), torch.zeros(k)
    rmsn = lambda Dv, b=zeros: dict(g=ones.clone(), b=c32(b), mode="rms", D=Dv)  # noqa: E731
    pw = g(T["patch_w"]).reshape(D, -1)
    Vj = V.T @ J
    W = dict(patch_w=c32((Vj @ pw).reshape(k, *T["patch_w"].shape[1:])), cls=c32(Vj @ g(T["cls"])),
             pos=c32(g(T["pos"]) @ Vj.T), ln_pre=dict(g=c32(S), b=c32(Ps[0] @ g(T["ln_pre"]["b"])), mode="rms", D=D * rho_e),
             blocks=[], skip=[])
    for i, B in enumerate(T["blocks"]):
        j1, j2, j3 = 2 * i, 2 * i + 1, 2 * i + 2
        A1 = torch.diag(g(B["ln1"]["g"])) @ Rs[j1]
        A2 = torch.diag(g(B["ln2"]["g"])) @ Rs[j2]
        b1, b2 = g(B["ln1"]["b"]), g(B["ln2"]["b"])
        nb_ = dict(ln1=rmsn(D * rho[j1]), ln2=rmsn(D * rho[j2]))
        for n in ("q", "k", "v"):
            nb_[f"{n}_w"] = c32(g(B[f"{n}_w"]) @ A1)
            nb_[f"{n}_b"] = c32(g(B[f"{n}_b"]) + g(B[f"{n}_w"]) @ b1)
        nb_["o_w"], nb_["o_b"] = c32(Ps[j2] @ g(B["o_w"])), c32(Ps[j2] @ g(B["o_b"]))
        nb_["fc_w"], nb_["fc_b"] = c32(g(B["fc_w"]) @ A2), c32(g(B["fc_b"]) + g(B["fc_w"]) @ b2)
        nb_["pr_w"], nb_["pr_b"] = c32(Ps[j3] @ g(B["pr_w"])), c32(Ps[j3] @ g(B["pr_b"]))
        W["blocks"].append(nb_)
        W["skip"].append((c32(Ps[j2] @ Rs[j1]), c32(Ps[j3] @ Rs[j2])))
    gp, bp = g(T["ln_post"]["g"]), g(T["ln_post"]["b"])
    Mo = Rs[-1].T @ torch.diag(gp) @ g(T["proj"])
    W["proj"] = c32(Mo)
    bpost = torch.linalg.lstsq(Mo.T, (bp @ g(T["proj"]))[:, None]).solution[:, 0]
    W["ln_post"] = rmsn(D * rho[-1], bpost)
    sk = [float((a - torch.eye(k)).norm() / k ** .5) for a, b in W["skip"]]
    print(f"[r3 stream ps] f={f:g} k={k} rho mean {sum(rho)/len(rho):.3f} min {min(rho):.3f} embed {rho_e:.3f} "
          f"post {rho[-1]:.3f} | skip-map distance from I (rms) mean {sum(sk)/len(sk):.3f}", flush=True)
    return W, ps.forward


@torch.no_grad()
def build_where(T, f, X, b):
    """Teacher at full width; stream projected onto global rG Q from block b on; norms rms with divisor D * rho_l."""
    D, nb = T["cls"].numel(), len(T["blocks"])
    k = vitgen.sizes(f, T)[0]
    Q, _ = core.basis(T, X, k, "rG")
    m = core.site_moments(T, X)
    rho = [float(torch.trace(Q.T @ m["M"][j] @ Q) / torch.trace(m["M"][j])) for j in range(2 * nb + 1)]
    J = torch.eye(D, dtype=dt) - 1.0 / D
    Pi = (J @ Q @ Q.T @ J).float()                       # = Q Q^T for a centred Q; = J at f = 1
    W = dict(T)
    W["_Pi"], W["_b"], W["_rho"], W["_D"] = Pi, b, rho, D
    print(f"[r3 stream where] f={f:g} k={k} b={b} rho from b: mean {sum(rho[2*b:])/max(1,len(rho[2*b:])):.3f}", flush=True)
    return W, fwd_where


def fwd_where(W, x):
    Pi, b, rho, D = W["_Pi"], W["_b"], W["_rho"], W["_D"]
    rmsd = lambda ln, j: dict(g=ln["g"], b=ln["b"], mode="rms", D=D * rho[j])  # noqa: E731
    s = vitgen.embed(W, x)
    for i, B in enumerate(W["blocks"]):
        if i < b:
            s = vitgen.block(s, B)
            continue
        if i == b:
            s = s @ Pi
        s = s + vitgen.attn(vitgen.norm(s, rmsd(B["ln1"], 2 * i)), B) @ Pi
        s = s + F.linear(F.gelu(F.linear(vitgen.norm(s, rmsd(B["ln2"], 2 * i + 1)), B["fc_w"], B["fc_b"])),
                         B["pr_w"], B["pr_b"]) @ Pi
    if b >= len(W["blocks"]):
        s = s - s.mean(-1, keepdim=True)
    return vitgen.norm(s[:, 0], rmsd(W["ln_post"], 2 * len(W["blocks"]))) @ W["proj"]


@torch.no_grad()
def score(W, fwd, split, stride, bs=100):
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "clipkd"))
    import eval_fast
    ref = eval_fast.teacher_ref(split)
    sel = np.arange(min(1000, len(ref["ids"])))[::stride]
    ref = dict(ref, ids=ref["ids"][sel], labels=ref["labels"][sel], logits=ref["logits"][sel])
    n = len(sel)
    X = eval_fast.photos(ref["ids"][:n])
    Wd = vitgen.to(W, board.DEV)
    if "skip" in W:
        Wd["skip"] = [(a.to(board.DEV), c.to(board.DEV)) for a, c in W["skip"]]
    cls = ref["cls"].to(board.DEV)
    lg = []
    for i in range(0, n, bs):
        e = fwd(Wd, X[i:i + bs].to(board.DEV).float()).float()
        lg.append((ref["scale"] * F.normalize(e, dim=-1) @ cls).cpu())
    lg = torch.cat(lg)
    zt = ref["logits"][:n]
    k = board.kl_rows(zt, lg)
    lab = ref["labels"][:n]
    m = (lg.argmax(1) == zt.argmax(1)).numpy()
    return dict(kl=float(k.mean()), top1=float((lg.argmax(1).numpy() == lab).mean()), agree=float(m.mean()), n=n), k.numpy(), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--f", default="0.85,0.7")
    ap.add_argument("--b", default="0,3,6,9")
    ap.add_argument("--input", required=True)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    T = board.clip_teacher()
    X = torch.load(a.input, weights_only=True).float()
    inp = os.path.basename(a.input)[:-3]
    stride = 1 if a.full else 5
    D = T["cls"].numel()
    for f in [float(v) for v in a.f.split(",")]:
        for b in ([int(v) for v in a.b.split(",")] if a.mode == "where" else [None]):
            t0 = time.time()
            if a.mode == "where":
                W, fwd = build_where(T, f, X, b)
                name = f"r3_stream_where_b{b}"
            else:
                k = vitgen.sizes(f, T)[0]
                W, fwd = build_ps(T, f, X, bases(T, X, k, a.mode, a.ridge, a.seed))
                name = f"r3_stream_ps_{a.mode}" + (f"_r{a.ridge:g}" if a.mode in ("ps", "derange") else "") + (f"_s{a.seed}" if a.seed else "")
            res, kl, mt = score(W, fwd, a.split, stride)
            row = dict(net="clip", rule=name, f=f, split=a.split, input=inp, **res, calc_seconds=round(time.time() - t0, 1),
                       photo_use=f"{a.split} photos, scoring only", when=datetime.datetime.now().isoformat(timespec="seconds"),
                       who="STREAM", quick=not a.full, note=a.note or "r3 STREAM r3_stream_ps.py")
            print(f"ROW {name} f={f} [{inp}] {'FULL' if a.full else 'quick'} {a.split} KL {res['kl']:.3f} agree {100*res['agree']:.1f}% "
                  f"top1 {100*res['top1']:.1f}%", flush=True)
            if a.full:
                tag = f"clip_{name}_f{f:g}_{a.split}{res['n']}_{inp}"
                np.save(os.path.join(HERE, "out", tag + ".npy"), kl)
                np.save(os.path.join(HERE, "out", tag + "_agree.npy"), mt)
                with open(os.path.join(HERE, "r3_stream_rows.jsonl"), "a") as fh:
                    fh.write(json.dumps(row) + "\n")
            else:
                with open(os.path.join(HERE, "scratch.jsonl"), "a") as fh:
                    fh.write(json.dumps(row) + "\n")
            del W
            if board.DEV == "mps":
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()
