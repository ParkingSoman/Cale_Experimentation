"""S2c (essay fix for the value side): rank value directions by what LATER neurons read, not by raw output size.

  python3 research/clean/liquid/s2c_vo_read.py --layers 12,16,22 --keep 0.5,0.25

Essay: a head is understood by "what it does when it finds it, in terms of neurons" (which neurons it affects).
Metric M = sum over readers after the head (layer L's MLP reads, all reads of layers > L) of W^T diag(e) W, with W the
read rows times the norm scale and e each row's pre-activation energy on calibration text; each matrix scaled to trace 1.
An unrestricted least-squares refit is unchanged by such a weighting, so only the rank choice (A) can change.
Arms: vo_svd vs vo_read (both nested, no refit), and their o_proj refits.
"""
import argparse
from lib import *  # noqa: F403
from lib import _capture  # noqa: E402
from s2_headsize import vo_bases, vo_apply, refit_o, AttnGuard, new_attn, lin  # noqa: E402
from s3_stream import collect, READS_IN, READS_POST  # noqa: E402


def down_metric(S, acc, L):
    M = np.zeros((D_MODEL, D_MODEL)); nl = len(S.model.model.layers)
    for l in range(L, nl):
        lay = S.model.model.layers[l]
        parts = [(READS_POST, lay.post_attention_layernorm.weight, lay.mlp, "h")]
        if l > L:
            parts.append((READS_IN, lay.input_layernorm.weight, lay.self_attn, "x"))
        for names, gam, par, key in parts:
            g = np.array(gam, dtype=np.float64); C = acc[(key, l)] / np.trace(acc[(key, l)])
            for n in names:
                W = np.array(getattr(par, n).weight, dtype=np.float64) * g
                e = np.einsum("ij,jk,ik->i", W, C, W); Gm = (W * e[:, None]).T @ W; M += Gm / np.trace(Gm)
    return M / np.trace(M) * D_MODEL


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="12,16,22"); ap.add_argument("--keep", default="0.5,0.25")
    a = ap.parse_args(); S = Setup(); acc = collect(S); rows = []
    for L in map(int, a.layers.split(",")):
        lay = S.model.model.layers[L]; at = lay.self_attn; t = Timer()
        Wv, Wo = (np.array(getattr(at, n).weight, dtype=np.float64) for n in ("v_proj", "o_proj"))
        z = _capture(S.model, S.cal, {"z": (at, "o_proj")})["z"]; Ya = (z.astype(np.float64) @ Wo.T).astype(np.float32)
        B0, B1 = vo_bases(z, Wo), vo_bases(z, Wo, down_metric(S, acc, L))
        for keep in map(float, a.keep.split(",")):
            r = int(round(64 * keep)); bf = (64 - r) * N_KV * 2304 / LAYER_PARAMS; res = {}
            for nm, B in (("vo_svd", B0), ("vo_read", B1)):
                Wv2, Wo2 = vo_apply(Wv, Wo, B, r)
                res[nm] = S.score([(lay, "self_attn", AttnGuard(new_attn(S, at, v_proj=Wv2, o_proj=Wo2), at))])
                Wls = refit_o(S, at, [("v_proj", lin(Wv2))], Ya, r)
                res[nm + "_ls"] = S.score([(lay, "self_attn", AttnGuard(new_attn(S, at, v_proj=Wv2, o_proj=Wls), at))])
            for nm, v in res.items():
                row = {"step": "S2c", "layer": L, "keep": keep, "arm": nm, "bytes_frac": round(bf, 4), **{k: boot(x) for k, x in v.items()}}
                rows.append(row); print(f"  L{L} keep{keep} {nm:11s} prose {row['prose'][0]:+.4f}  code {row['code'][0]:+.4f}", flush=True)
            for x, y in (("vo_read", "vo_svd"), ("vo_read_ls", "vo_svd_ls")):
                d = {k: boot(res[x][k] - res[y][k]) for k in res[x]}
                rows.append({"step": "S2c", "layer": L, "keep": keep, "diff": f"{x} - {y}", **d})
                print(f"  L{L} keep{keep} {x} - {y}: prose {np.round(d['prose'], 4).tolist()} code {np.round(d['code'], 4).tolist()}", flush=True)
        print(f"L{L} {t()}s", flush=True)
    print("saved", save("s2c-vo-read", rows))


if __name__ == "__main__":
    main()
