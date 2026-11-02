"""S2: head-size dial. Can each head shrink in small nested steps instead of whole heads?

  python3 research/clean/liquid/s2_headsize.py --layers 12,16,22

Two sides of a head, each with a data arm and a control (all with the position-0 sink guard):
  VO side (value dims, one basis per key/value group because its 3 query heads share the values):
    vo_svd      whitened SVD order: A = V_r^T C^-1/2, B_h = Wo_h C^1/2 V_r, with C = pooled head-output covariance and
                V_r = top right singular vectors of [Wo_1; Wo_2; Wo_3] C^1/2. Nested by construction (no refit).
    vo_svd_ls   same A, then CV-ridge refit of all kept o_proj columns (not nested across sizes)
    vo_norm_ls  control: keep value dims by (output std x o-column norm), then the same refit
  QK side (rotary pairs (i, i+32) dropped whole, shared by the group's query heads and its key head):
    qk_act      drop pairs with the smallest sum_h E|q_h,pair| * E|k_pair| on calibration text (MHA2MLA-style 2-norm)
    qk_wnorm    control: same score from weight norms only
  both          qk_act + vo_svd_ls at the same keep fraction (VO refit recaptured under the new attention)
Removal is emulated by zero rows/columns; bytes are counted as removed. One head dim (per group) = 2,304 weights on
either side, so a whole query head = 32 dims.
"""
import argparse
from lib import *  # noqa: F403
from lib import _capture, _cv_ridge  # noqa: E402

G = 3
HALF = HEAD_DIM // 2


class AttnGuard(nn.Module):
    """New attention everywhere except block position 0, which gets the original block's output (position 0 attends
    only to itself, so the original block run on the first word alone is exact)."""
    def __init__(self, new, orig):
        super().__init__(); self.new, self.orig = new, orig
    def __call__(self, x, mask=None, cache=None):
        y = self.new(x, mask, cache)
        return mx.concatenate([self.orig(x[:, :1], None, None), y[:, 1:]], axis=1)


def new_attn(S, at, **W):
    from mlx_lm.models.llama import Attention
    a = Attention(S.model.args); a.update(at.parameters())
    for n, w in W.items():
        setattr(a, n, lin(w))
    return a


def lin(W):
    m = nn.Linear(W.shape[1], W.shape[0], bias=False); m.weight = mx.array(W.astype(np.float32)); return m


def psd_pow(C, p, eps=1e-6):
    w, V = np.linalg.eigh(C); w = np.maximum(w, eps * w.max()); return (V * w ** p) @ V.T


def vo_bases(z, Wo, M=None):
    """Per group: whitened-SVD A (64x64, rows in importance order) and B_h (576x64 per head).
    M (optional, 576x576): metric of what later neurons read; the SVD then ranks by read-weighted output (S2c)."""
    Mh = psd_pow(M, .5) if M is not None else np.eye(Wo.shape[0])
    out = []
    for g in range(N_KV):
        hs = range(g * G, (g + 1) * G)
        Zs = [z[:, h * HEAD_DIM:(h + 1) * HEAD_DIM].astype(np.float64) for h in hs]
        C = sum(Zi.T @ Zi for Zi in Zs) / (len(Zs) * len(Zs[0]))
        Ch, Cih = psd_pow(C, .5), psd_pow(C, -.5)
        Wst = np.concatenate([Mh @ Wo[:, h * HEAD_DIM:(h + 1) * HEAD_DIM] for h in hs], 0) @ Ch
        _, _, Vt = np.linalg.svd(Wst, full_matrices=False)
        A = Vt @ Cih
        Bs = [Wo[:, h * HEAD_DIM:(h + 1) * HEAD_DIM] @ Ch @ Vt.T for h in hs]
        out.append((A, Bs))
    return out


def vo_apply(Wv, Wo, bases, r):
    """New v_proj / o_proj with r value dims per group (rest zero). Returns (Wv_new, Wo_new)."""
    Wv2 = np.zeros_like(Wv); Wo2 = np.zeros_like(Wo)
    for g, (A, Bs) in enumerate(bases):
        Wv2[g * HEAD_DIM:g * HEAD_DIM + r] = A[:r] @ Wv[g * HEAD_DIM:(g + 1) * HEAD_DIM]
        for j, h in enumerate(range(g * G, (g + 1) * G)):
            Wo2[:, h * HEAD_DIM:h * HEAD_DIM + r] = Bs[j][:, :r]
    return Wv2, Wo2


def kept_cols(r):
    return np.concatenate([np.arange(h * HEAD_DIM, h * HEAD_DIM + r) for h in range(N_HEADS)])


def refit_o(S, at, edits_qkv, Ya, r):
    """Recapture head outputs under edits (q/k/v), CV-ridge refit of the kept o columns."""
    old = [(a, getattr(at, a)) for a, _ in edits_qkv]
    for a, m in edits_qkv:
        setattr(at, a, m)
    z = _capture(S.model, S.cal, {"z": (at, "o_proj")})["z"]
    for a, m in old:
        setattr(at, a, m)
    kc = kept_cols(r); W = np.zeros((D_MODEL, N_HEADS * HEAD_DIM)); W[:, kc] = _cv_ridge(z[:, kc], Ya).T
    return W


def qk_drop(Wq, Wk, score, n_keep):
    """score[g, i] for pair i in group g. Zero the weakest pairs' rows in q (all group heads) and k."""
    Wq2, Wk2 = Wq.copy(), Wk.copy()
    for g in range(N_KV):
        drop = np.argsort(score[g])[:HALF - n_keep]
        rows = np.concatenate([drop, drop + HALF])
        Wk2[g * HEAD_DIM + rows] = 0
        for h in range(g * G, (g + 1) * G):
            Wq2[h * HEAD_DIM + rows] = 0
    return Wq2, Wk2


def pair_norm(Y, nh):
    """Mean 2-norm of each rotary pair per head: Y (n, nh*64) -> (nh, 32)."""
    Y = Y.reshape(len(Y), nh, HEAD_DIM)
    return np.sqrt(Y[..., :HALF] ** 2 + Y[..., HALF:] ** 2).mean(0)


def run_layer(S, L, keeps, rows):
    lay = S.model.model.layers[L]; at = lay.self_attn; t = Timer()
    Wq, Wk, Wv, Wo = (np.array(getattr(at, n).weight, dtype=np.float64) for n in ("q_proj", "k_proj", "v_proj", "o_proj"))
    cap = _capture(S.model, S.cal, {"z": (at, "o_proj"), "xa": (at, "q_proj")})
    z = cap["z"]; xa = cap["xa"][::4].astype(np.float64)
    Ya = (z.astype(np.float64) @ Wo.T).astype(np.float32)
    bases = vo_bases(z, Wo)
    qn, kn = pair_norm(xa @ Wq.T, N_HEADS), pair_norm(xa @ Wk.T, N_KV)
    s_act = np.array([sum(qn[h] for h in range(g * G, (g + 1) * G)) * kn[g] for g in range(N_KV)])
    wqn, wkn = pair_norm(Wq.T, N_HEADS), pair_norm(Wk.T, N_KV)  # rows of W.T = input coords; norm over inputs
    s_w = np.array([sum(wqn[h] for h in range(g * G, (g + 1) * G)) * wkn[g] for g in range(N_KV)])
    zstd = z.std(0)
    print(f"L{L} capture+bases {t()}s", flush=True)
    for keep in keeps:
        r = int(round(HEAD_DIM * keep)); npair = r // 2
        dims_removed = (HEAD_DIM - r) * N_KV  # per side, summed over groups
        bf_side = dims_removed * 2304 / LAYER_PARAMS
        arms = {}
        Wv2, Wo2 = vo_apply(Wv, Wo, bases, r)
        arms["vo_svd"] = ([(lay, "self_attn", AttnGuard(new_attn(S, at, v_proj=Wv2, o_proj=Wo2), at))], bf_side)
        Wo_ls = refit_o(S, at, [("v_proj", lin(Wv2))], Ya, r)
        arms["vo_svd_ls"] = ([(lay, "self_attn", AttnGuard(new_attn(S, at, v_proj=Wv2, o_proj=Wo_ls), at))], bf_side)
        # control: keep value dims by output std x o-column norm (per group, shared by its heads), permuted to the front
        Wv3 = np.zeros_like(Wv)
        for g in range(N_KV):
            hs = range(g * G, (g + 1) * G)
            sc = sum(zstd[h * HEAD_DIM:(h + 1) * HEAD_DIM] * np.linalg.norm(Wo[:, h * HEAD_DIM:(h + 1) * HEAD_DIM], axis=0) for h in hs)
            idx = np.argsort(-sc)[:r]
            Wv3[g * HEAD_DIM:g * HEAD_DIM + r] = Wv[g * HEAD_DIM + idx]
        Wo3 = refit_o(S, at, [("v_proj", lin(Wv3))], Ya, r)
        arms["vo_norm_ls"] = ([(lay, "self_attn", AttnGuard(new_attn(S, at, v_proj=Wv3, o_proj=Wo3), at))], bf_side)
        for nm, sc in (("qk_act", s_act), ("qk_wnorm", s_w)):
            Wq2, Wk2 = qk_drop(Wq, Wk, sc, npair)
            arms[nm] = ([(lay, "self_attn", AttnGuard(new_attn(S, at, q_proj=Wq2, k_proj=Wk2), at))], bf_side)
        Wq2, Wk2 = qk_drop(Wq, Wk, s_act, npair)
        Wo4 = refit_o(S, at, [("q_proj", lin(Wq2)), ("k_proj", lin(Wk2)), ("v_proj", lin(Wv2))], Ya, r)
        arms["both"] = ([(lay, "self_attn", AttnGuard(new_attn(S, at, q_proj=Wq2, k_proj=Wk2, v_proj=Wv2, o_proj=Wo4), at))], 2 * bf_side)
        print(f"L{L} keep{keep} fits {t()}s", flush=True)
        res = {}
        for name, (edits, bf) in arms.items():
            res[name] = S.score(edits)
            row = {"step": "S2", "layer": L, "keep": keep, "arm": name, "bytes_frac": round(bf, 4),
                   **{k: boot(v) for k, v in res[name].items()}}
            rows.append(row)
            print(f"  L{L} keep{keep} {name:11s} bytes {bf:.4f}  prose {row['prose'][0]:+.4f}  code {row['code'][0]:+.4f}", flush=True)
        for a, b in (("vo_svd_ls", "vo_norm_ls"), ("qk_act", "qk_wnorm"), ("vo_svd", "vo_svd_ls")):
            d = {k: boot(res[a][k] - res[b][k]) for k in res[a]}
            rows.append({"step": "S2", "layer": L, "keep": keep, "diff": f"{a} - {b}", **d})
            print(f"  L{L} keep{keep} {a} - {b}: prose {np.round(d['prose'], 4).tolist()} code {np.round(d['code'], 4).tolist()}", flush=True)
        print(f"L{L} keep{keep} scored {t()}s", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="12,16,22"); ap.add_argument("--keep", default="0.75,0.5,0.25")
    a = ap.parse_args(); S = Setup(); rows = []
    for L in map(int, a.layers.split(",")):
        run_layer(S, L, list(map(float, a.keep.split(","))), rows)
        print("saved", save("s2-headsize", rows)); rows.clear()


if __name__ == "__main__":
    main()
