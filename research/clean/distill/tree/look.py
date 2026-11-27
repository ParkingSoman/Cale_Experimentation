# scale: neurons=every node of the built networks; net=both; photos=1,000 ResNet test / 200 CLIP test photos, reading only; where=mac
# rung: Q=tree loop look (stream error by block, firing breadth vs trained students, named branch s3:69); EST=3 min; PERMIT=Shonu approved plan 2026-11-26 late night
"""The look step (PAGE.md, plan step 4), read whole. Photo use: reading only (test photos). No calculation uses them.

r18:  per block, relative stream error of the built network vs A (block outputs), and the inner firing rate
      (share of photo x cell > 0, anskey.inner_fire) of the built network vs A vs the trained-from-random student of
      the same size (answer key: anskey_rate_W95R_s0 / W50R_s0 / W5R_s0_x4). Named branch: s3:69's 11 R1 children
      (CALC 3.7) in s4b0 = tree block 6: their main parents, their coarse groups, and the share of each child's top 1%
      cells where its coarse node fires.
clip: per block, relative stream error (all tokens) and MLP firing rate (share of tokens with pre-activation > 0).
Usage: curves/.venv/bin/python -I distill/tree/look.py r18|clip [--rule tree_v1] [--f 0.95,0.85,0.5,0.05]
Output: distill/tree/out/look_<net>_<rule>.json
"""
import argparse
import importlib
import json
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.dirname(HERE)
FAN = os.path.join(DIST, "fanout")
sys.path.insert(0, HERE)
sys.path.insert(0, FAN)
sys.path.insert(0, DIST)
import board  # noqa: E402
import build as TB  # noqa: E402

CHILDREN = [305, 97, 23, 163, 385, 297, 464, 237, 108, 510, 275]     # CALC 3.7, R1 children of s3:69 in s4b0
RKEY = {0.95: "W95R_s0", 0.5: "W50R_s0", 0.05: "W5R_s0_x4"}


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


@torch.no_grad()
def look_r18(rule, fs):
    import anskey
    import train_whole as TW
    ph = TW.r18.net.Photos("read")
    test = TW.split(ph.lab)["test"]
    rows = test[::max(1, len(test) // 1000)][:1000]
    x = TW.photos(ph, rows)
    A = board.r18_A().to(TB.DEV).eval()
    fA = anskey.inner_fire(A, x)

    def outs(m):
        o = []
        hs = [blk.register_forward_hook(lambda mod, i, out: o.append(out.float().cpu())) for blk in TB.r18_blocks(m)]
        for a in range(0, 200, 50):
            m(x[a:a + 50].float())
        for h in hs:
            h.remove()
        nb = len(hs)
        return [torch.cat(o[b::nb]) for b in range(nb)]
    oA = outs(A)
    res = []
    for f in fs:
        S = rule.r18(A.cpu(), f).to(TB.DEV).eval()
        groups = S._infos
        fS = anskey.inner_fire(S, x)
        oS = outs(S)
        names = [(s, b) for s in range(4) for b in range(2)]
        rk = RKEY.get(f)
        ans = json.load(open(os.path.join(FAN, "out", f"anskey_rate_{rk}.json"))) if rk else None
        blocks = []
        for i, k in enumerate(names):
            row = dict(block=f"s{k[0] + 1}b{k[1]}", stream_err=round(rel(oS[i], oA[i]), 4),
                       rate_A=round(float(fA[k]["rate"].mean()), 3), rate_built=round(float(fS[k]["rate"].mean()), 3),
                       n_built=int(fS[k]["rate"].numel()))
            if ans:
                row["rate_R_student"] = round(ans[i]["S"], 3)
            blocks.append(row)
        # named branch: s3:69's children live in tree block 6 (s4b0)
        _tree = sys.modules["rules._tree"]
        st, tr = list(_tree._C.values())[-1]
        grp, _ = TB.r18_groups(A.cpu(), tr, f, loo=getattr(rule, "LOO", None))
        g6 = grp[6]
        where = {j: gi for gi, M in enumerate(g6) for j in M}
        Ach = fA[(3, 0)]["mx"]
        named = []
        for c in CHILDREN:
            gi = where[c]
            p = tr["parent"][tr["gid"][("m", 6, c)]]
            pname = tr["names"][p]
            hit = None
            if len(g6[gi]) > 1:
                mxA, mxS = fA[(3, 0)]["mx"][:, c], fS[(3, 0)]["mx"][:, gi]
                top = mxA >= mxA.quantile(0.99)
                hit = round(float((mxS[top] > 0).float().mean()), 3)
            named.append(dict(child=c, main_parent=f"{pname[0]}{'' if pname[0] == 'e' else pname[1]}:{pname[-1]}",
                              group_size=len(g6[gi]), siblings_in_group=sorted(set(g6[gi]) & set(CHILDREN) - {c}),
                              coarse_fires_on_child_top1pct_photos=hit))
        del Ach
        res.append(dict(f=f, blocks=blocks, named_s3_69=named))
        print(json.dumps(res[-1]), flush=True)
        del S
    return res


@torch.no_grad()
def look_clip(rule, fs):
    sys.path.insert(0, os.path.join(DIST, "clipkd"))
    import eval_fast
    import vitgen
    ref = eval_fast.teacher_ref("test")
    x = eval_fast.photos(ref["ids"][:1000][::5])
    T = board.clip_teacher()

    def run(W):
        Wd = vitgen.to(W, TB.DEV)
        st, rates = [], []
        for a in range(0, len(x), 25):
            s = vitgen.embed(Wd, x[a:a + 25].to(TB.DEV).float())
            ss, rr = [], []
            for B in Wd["blocks"]:
                s = s + vitgen.attn(vitgen.norm(s, B["ln1"]), B)
                z = F.linear(vitgen.norm(s, B["ln2"]), B["fc_w"], B["fc_b"])
                rr.append(float((z > 0).float().mean()))
                s = s + F.linear(F.gelu(z), B["pr_w"], B["pr_b"])
                ss.append(s.cpu())
            st.append(ss)
            rates.append(rr)
        nb = len(Wd["blocks"])
        return [torch.cat([b[i] for b in st]) for i in range(nb)], [sum(r[i] for r in rates) / len(rates) for i in range(nb)]
    sT, rT = run(T)
    res = []
    for f in fs:
        W = rule.vit(T, f)
        sW, rW = run(W)
        blocks = [dict(block=i, stream_err=round(rel(sW[i], sT[i]), 4), mlp_rate_big=round(rT[i], 3), mlp_rate_built=round(rW[i], 3),
                       mlp_n=W["blocks"][i]["fc_w"].shape[0], heads=W["blocks"][i]["q_w"].shape[0] // 64) for i in range(len(sT))]
        res.append(dict(f=f, blocks=blocks))
        print(json.dumps(res[-1]), flush=True)
        del W
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net", choices=("r18", "clip"))
    ap.add_argument("--rule", default="tree_v1")
    ap.add_argument("--f", default="0.95,0.85,0.5,0.05")
    a = ap.parse_args()
    rule = importlib.import_module(f"rules.{a.rule}")
    fs = [float(v) for v in a.f.split(",")]
    res = look_r18(rule, fs) if a.net == "r18" else look_clip(rule, fs)
    out = os.path.join(HERE, "out", f"look_{a.net}_{a.rule}.json")
    json.dump(dict(net=a.net, rule=a.rule, photo_use="reading only (test photos)", sizes=res), open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
