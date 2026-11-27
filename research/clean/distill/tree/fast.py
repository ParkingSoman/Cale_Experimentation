# scale: neurons=every node; net=clip|r18; photos=200 test (CLIP quick) or 1,000 (ResNet), scoring only; where=mac
# rung: Q=fast loop runner (stats once per process, many rows); EST=CLIP 70 s + 15 s per row, ResNet 5 s per row; PERMIT=Shonu approved plan 2026-11-26 late night
"""Fast loop runner: one process, statistics and tree computed once, many rows.
rows: name:f[:loo][:parts]   parts = hm (default), h (heads coarse only), m (MLP coarse only)  -- parts != hm is a BUG HUNT row only
Usage: curves/.venv/bin/python -I distill/tree/fast.py clip "a:0.95" "b:0.95::h" ...   (CLIP scores on the quick 200; --full for 1,000)
Rows append to distill/tree/out/fast.jsonl."""
import json, os, sys, time, datetime
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); DIST = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(DIST, "fanout"))
import board, build as TB, vitgen  # noqa: E402

net = sys.argv[1]; full = "--full" in sys.argv
rows = [a for a in sys.argv[2:] if not a.startswith("--")]
X = torch.load(os.path.join(DIST, "fanout", "cache", f"{net}_self_n256_k32_s1.pt"), weights_only=True).float()
big = board.clip_teacher() if net == "clip" else board.r18_A().cpu()
t0 = time.time()
# statistics + tree cached on disk, keyed by the source of the code that makes them (a change there remakes them)
import hashlib, inspect  # noqa: E402
fns = [TB.clip_stats, TB.clip_tree, TB.heads_out, TB._centre] if net == "clip" else [TB.r18_stats, TB.r18_tree, TB._bn_affine, TB.r18_blocks]
key = hashlib.md5("".join(inspect.getsource(g) for g in fns + [TB.chains]).encode()).hexdigest()[:10]
cp = os.path.join(HERE, "out", f"cache_{net}_{key}.pt")
if os.path.exists(cp):
    d = torch.load(cp, weights_only=False); st, tr = d["st"], d["tr"]
else:
    st = (TB.clip_stats if net == "clip" else TB.r18_stats)(big, X)
    tr = (TB.clip_tree if net == "clip" else TB.r18_tree)(big, st)
    torch.save(dict(st=st, tr=tr), cp)
print(f"stats+tree {time.time() - t0:.0f} s ({os.path.basename(cp)})", flush=True)


def overlap(f, loo):
    """LOOK row: does each coarse node fire on the UNION of its members (generic input, big network's own stream)?
    prec = P(union | coarse fires); rec = P(coarse fires | union); srec = P(coarse fires | some member in its top 1%)."""
    import torch.nn.functional as F
    if net == "clip":
        groups = [g["m"] for g in TB.clip_groups(big, tr, f, loo)[0]]; act = F.gelu
    else:
        groups = TB.r18_groups(big, tr, f, loo)[0]; act = F.relu
    out = []
    for b, G in enumerate(groups):
        Z = st["Z"][b].float(); lvl = TB.strong(Z)
        thr = Z.topk(max(1, Z.shape[0] // 100), dim=0).values[-1]
        P, R, SR, RT = [], [], [], []
        for M in G:
            if len(M) == 1: continue
            rep = max(M, key=lambda j: float(tr["strength"][("m", b)][j]))
            c, b0, w = TB.coarse_unit(Z, M, lvl, act, loo, rep)
            fc = (Z[:, M] @ c + b0) > 0; u = (Z[:, M] > 0).any(1); su = (Z[:, M] >= thr[M]).any(1)
            P.append(float((fc & u).sum() / fc.sum().clamp_min(1))); R.append(float((fc & u).sum() / u.sum().clamp_min(1)))
            SR.append(float((fc & su).sum() / su.sum().clamp_min(1))); RT.append(float(fc.float().mean() / u.float().mean().clamp_min(1e-6)))
        med = lambda v: round(float(torch.tensor(v).median()), 3) if v else None  # noqa: E731
        out.append(dict(block=b, groups=len(P), prec=med(P), rec=med(R), srec=med(SR), rate_ratio=med(RT)))
    return out


for spec in rows:
    p = spec.split(":") + ["", "", ""]
    name, f, loo, parts = p[0], float(p[1]), (p[2] or None), (p[3] or "hm")
    t1 = time.time()
    TB.COARSE_PARENT = "P" in parts; TB.WRITE_PARTNER = "W" in parts
    if name.startswith("ov"):
        o = overlap(f, loo)
        print(json.dumps(dict(net=net, look="overlap", f=f, loo=loo, blocks=o)), flush=True)
        open(os.path.join(HERE, "out", "fast.jsonl"), "a").write(json.dumps(dict(net=net, look="overlap", f=f, loo=loo, blocks=o)) + "\n")
        continue
    if net == "clip":
        _, Hk, Mk = vitgen.sizes(f, big)
        ks = {}
        for b in range(len(big["blocks"])):
            ks[("h", b)] = Hk if "h" in parts else 12
            ks[("m", b)] = Mk if "m" in parts else big["blocks"][b]["fc_w"].shape[0]
        if "L" in parts:                                       # Shonu's cut rule: one level for every block
            nb = len(big["blocks"])
            kh, Lh = TB.level_ks(tr, [("h", b) for b in range(nb)], f); km, Lm = TB.level_ks(tr, [("m", b) for b in range(nb)], f)
            ks = {**kh, **km}; print(f"  level heads {Lh} mlp {Lm} | mlp widths {[km[('m', b)] for b in range(nb)]}", flush=True)
        W = TB.build_clip(big, f, X, loo=loo, st=st, tr=tr, ks=ks, verbose=False); W.pop("_infos")
        res, kk, mm = board.clip_score(W, 1000, "test", stride=1 if full else 5, with_match=True); params = vitgen.n_params(W)
    else:
        ks = None
        if "L" in parts:
            ks, Lr = TB.level_ks(tr, [("m", b) for b in range(tr["nb"])], f); print(f"  level {Lr} | widths {[ks[('m', b)] for b in range(tr['nb'])]}", flush=True)
        S = TB.build_r18(big.cpu(), f, X, loo=loo, st=st, tr=tr, verbose=False, ks=ks)
        res, kk, mm = board.r18_score(S, 1000, "test", with_match=True); params = sum(q.numel() for q in S.parameters())
    if full or net == "r18":                                  # per-photo files for the 800-unseen read (r3_critic_holdout.py)
        import numpy as np
        tag = f"tree_{net}_{name}"
        np.save(os.path.join(DIST, "fanout", "out", tag + ".npy"), kk); np.save(os.path.join(DIST, "fanout", "out", tag + "_agree.npy"), mm)
    row = dict(net=net, name=name, f=f, loo=loo, parts=parts, **res, params=params, sec=round(time.time() - t1, 1),
               when=datetime.datetime.now().isoformat(timespec="seconds"))
    print(json.dumps(row), flush=True)
    open(os.path.join(HERE, "out", "fast.jsonl"), "a").write(json.dumps(row) + "\n")
