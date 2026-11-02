"""S3: stream width at one layer. How much of one layer's reads and writes fit in a smaller fixed stream subspace?

  python3 research/clean/liquid/s3_stream.py --layers 12,16,22

Exact emulation of "fold norm scales, rotate the whole model by an orthogonal basis Q, slice layer L to d' dims":
  reads  (q, k, v with gamma_in; gate, up with gamma_post):  W' = W diag(gamma) P P^T diag(1/gamma)
  writes (o_proj, down_proj):                                 W' = P P^T W
P = the first d' columns of the basis. Every matrix of a layer has one 576-wide side, so slicing to d' removes
(1 - d'/576) of the layer's weights. The norms stay as they are (a full-width rms is one number per word).
Bases (stream statistics exclude block position 0; x_hat = stream / rms(stream), i.e. after the norm, before gamma):
  pca_global  eigvectors of sum over all 60 norm inputs of E[x_hat x_hat^T] (SliceGPT style, one basis for all layers)
  write_glob  eigvectors of the pooled covariance of what layers WRITE (attention + MLP outputs / stream rms)
  read_glob   eigvectors of the pooled read-weight Gram diag(gamma) W^T W diag(gamma), each matrix scaled to trace 1
  coord       control: raw coordinates ordered by pooled E[x_hat_i^2]
  pca_local   reference, NOT free: eigvectors of layer L's own two norm inputs (needs its own stored basis)
Saliency coordinates (pooled E[x_hat_i^2] > 20x median) are put first in every basis.
_ls variants: least-squares read refit W' = W diag(gamma) C P (P^T C P)^-1 P^T with C = layer L's own input stats.
Equal-bytes MLP references: cut_ls_mix and rcf_v1 removing the same share of the layer's weights.
"""
import argparse
from lib import *  # noqa: F403
from contract import load_method  # noqa: E402

READS_IN, READS_POST, WRITES = ("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj"), (("self_attn", "o_proj"), ("mlp", "down_proj"))


class StatBlock(nn.Module):
    """Wraps a TransformerBlock, runs its forward and accumulates normalized stream / write Gram matrices."""

    def __init__(self, blk, acc, l):
        super().__init__(); self.blk, self.acc, self.l = blk, acc, l
        self.use_sliding = getattr(blk, "use_sliding", False)

    def __call__(self, x, mask=None, cache=None):
        b = self.blk
        r = b.self_attn(b.input_layernorm(x), mask, cache); h = x + r
        r2 = b.mlp(b.post_attention_layernorm(h)); out = h + r2
        def nrm(v, s):
            return (v / s)[:, 1:].reshape(-1, v.shape[-1])
        sx = mx.sqrt(mx.mean(x * x, -1, keepdims=True) + 1e-5); sh = mx.sqrt(mx.mean(h * h, -1, keepdims=True) + 1e-5)
        for key, v in (("x", nrm(x, sx)), ("h", nrm(h, sh)), ("wa", nrm(r, sx)), ("wm", nrm(r2, sh))):
            g = np.array(v.T @ v, dtype=np.float64); self.acc[(key, self.l)] = self.acc.get((key, self.l), 0) + g
        return out



def collect(S, n_blocks=32):
    acc = {}; layers = S.model.model.layers; orig = list(layers)
    for l in range(len(orig)):
        layers[l] = StatBlock(orig[l], acc, l)
    blocks = mx.concatenate([S.cal[:n_blocks // 2], S.cal[-n_blocks // 2:]], 0)
    for i in range(0, len(blocks), 4):
        mx.eval(S.model(blocks[i:i + 4, :-1]))
    for l in range(len(orig)):
        layers[l] = orig[l]
    return acc


def eig_desc(M):
    w, V = np.linalg.eigh(M); return V[:, ::-1]


def with_saliency(B, sal):
    """Put the saliency coordinate axes first, then B's columns with those axes projected out (re-orthonormalized)."""
    if len(sal) == 0:
        return B
    E = np.eye(D_MODEL)[:, sal]
    R = B - E @ (E.T @ B)
    Q, _ = np.linalg.qr(np.concatenate([E, R], 1))
    Q[:, :len(sal)] = E  # keep signs exact
    return Q[:, :D_MODEL]


def bases(S, acc, L):
    nl = len(S.model.model.layers)
    Cx = sum(acc[("x", l)] + acc[("h", l)] for l in range(nl))
    diag = np.diag(Cx); sal = list(np.where(diag > 20 * np.median(diag))[0])
    Cw = sum(acc[("wa", l)] + acc[("wm", l)] for l in range(nl))
    M = np.zeros((D_MODEL, D_MODEL))
    for l in range(nl):
        lay = S.model.model.layers[l]
        for names, gam, par in ((READS_IN, lay.input_layernorm.weight, lay.self_attn), (READS_POST, lay.post_attention_layernorm.weight, lay.mlp)):
            g = np.array(gam, dtype=np.float64)
            for n in names:
                W = np.array(getattr(par, n).weight, dtype=np.float64) * g
                Gm = W.T @ W; M += Gm / np.trace(Gm)
    # essay fix (S3b): "the directions neurons write to and read from", weighted by how active each reader is
    Ma = np.zeros((D_MODEL, D_MODEL))
    for l in range(nl):
        lay = S.model.model.layers[l]
        for names, gam, par, key in ((READS_IN, lay.input_layernorm.weight, lay.self_attn, "x"), (READS_POST, lay.post_attention_layernorm.weight, lay.mlp, "h")):
            g = np.array(gam, dtype=np.float64); C = acc[(key, l)] / np.trace(acc[(key, l)])
            for n in names:
                W = np.array(getattr(par, n).weight, dtype=np.float64) * g
                e = np.einsum("ij,jk,ik->i", W, C, W)  # pre-activation energy of each reading row
                Gm = (W * e[:, None]).T @ W; Ma += Gm / np.trace(Gm)
    B = {"pca_global": eig_desc(Cx), "write_glob": eig_desc(Cw), "read_glob": eig_desc(M),
         "read_act": eig_desc(Ma), "rw_act": eig_desc(Ma / np.trace(Ma) + Cw / np.trace(Cw)),
         "coord": np.eye(D_MODEL)[:, np.argsort(-diag)],
         "pca_local": eig_desc(acc[("x", L)] / np.trace(acc[("x", L)]) + acc[("h", L)] / np.trace(acc[("h", L)]))}
    return {k: with_saliency(v, sal) if k != "coord" else v for k, v in B.items()}, sal


def sliced_layer(S, L, P, acc=None):
    """Edits replacing layer L's reads/writes by their d'-subspace versions. acc given -> LS read refit."""
    lay = S.model.model.layers[L]; edits = []; PP = P @ P.T
    for names, gam, par, key in ((READS_IN, lay.input_layernorm.weight, lay.self_attn, "x"), (READS_POST, lay.post_attention_layernorm.weight, lay.mlp, "h")):
        g = np.array(gam, dtype=np.float64)
        if acc is not None:
            C = acc[(key, L)]; proj = C @ P @ np.linalg.solve(P.T @ C @ P, P.T)  # x_hat -> LS estimate from P^T x_hat
        else:
            proj = PP
        for n in names:
            W = np.array(getattr(par, n).weight, dtype=np.float64) * g
            Wn = (W @ proj) / g
            m = nn.Linear(D_MODEL, W.shape[0], bias=False); m.weight = mx.array(Wn.astype(np.float32)); edits.append((par, n, m))
    for pn, n in WRITES:
        par = getattr(lay, pn); W = np.array(getattr(par, n).weight, dtype=np.float64)
        m = nn.Linear(W.shape[1], D_MODEL, bias=False); m.weight = mx.array((PP @ W).astype(np.float32)); edits.append((par, n, m))
    return edits


def run_layer(S, acc, L, dps, rows, only=None, mlp=True, ls=("pca_global", "write_glob", "read_glob")):
    t = Timer(); B, sal = bases(S, acc, L); lay = S.model.model.layers[L]
    if only:
        B = {k: v for k, v in B.items() if k in only}
    print(f"L{L} bases {t()}s  saliency coords {sal}", flush=True)
    for dp in dps:
        frac = 1 - dp / D_MODEL; res = {}
        arms = {k: sliced_layer(S, L, B[k][:, :dp]) for k in B}
        for k in ls:
            if k in B:
                arms[k + "_ls"] = sliced_layer(S, L, B[k][:, :dp], acc)
        K = D_FF - int(round(frac * LAYER_PARAMS / NEURON_PARAMS))
        for name in (("cut_ls_mix", "rcf_v1") if mlp else ()):
            mod = load_method(os.path.join(CLEAN, "coarse", "methods", f"{name}.py"))
            layer = mod.build(S.model, S.tok, L, K, S.cal[:64], log=lambda *a: None) if name == "rcf_v1" else mod.build(S.model, S.tok, L, K, S.cal[:64])
            arms[f"mlp_{name}"] = [(lay, "mlp", layer)] + list(getattr(layer, "extra", []))
        for name, edits in arms.items():
            res[name] = S.score(edits)
            row = {"step": "S3", "layer": L, "d": dp, "arm": name, "bytes_frac": round(frac, 4), "mlp_K": K if name.startswith("mlp") else None,
                   **{k: boot(v) for k, v in res[name].items()}}
            rows.append(row)
            print(f"  L{L} d{dp} {name:15s} prose {row['prose'][0]:+.4f}  code {row['code'][0]:+.4f}", flush=True)
        best = min((k for k in arms if not k.startswith("mlp") and k != "pca_local"), key=lambda k: res[k]["prose"].mean() + res[k]["code"].mean())
        pairs = [(best, "mlp_rcf_v1"), (best, "coord"), ("pca_global", "pca_local")] if mlp else [(a, b) for a in res for b in ("read_glob_ls", "pca_global") if a != b and b in res and a.startswith(("read_act", "rw_act"))]
        for a, b in pairs:
            d = {k: boot(res[a][k] - res[b][k]) for k in res[a]}
            rows.append({"step": "S3", "layer": L, "d": dp, "diff": f"{a} - {b}", **d})
            print(f"  L{L} d{dp} {a} - {b}: prose {np.round(d['prose'], 4).tolist()} code {np.round(d['code'], 4).tolist()}", flush=True)
        print(f"L{L} d{dp} scored {t()}s", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="12,16,22"); ap.add_argument("--d", default="504,432,288")
    ap.add_argument("--only", default=""); ap.add_argument("--no-mlp", action="store_true"); ap.add_argument("--ls", default="pca_global,write_glob,read_glob"); ap.add_argument("--tag", default="s3-stream")
    a = ap.parse_args(); S = Setup(); t = Timer(); acc = collect(S); print(f"stats {t()}s", flush=True); rows = []
    for L in map(int, a.layers.split(",")):
        run_layer(S, acc, L, list(map(int, a.d.split(","))), rows, a.only.split(",") if a.only else None, not a.no_mlp, tuple(a.ls.split(",")))
        print("saved", save(a.tag, rows)); rows.clear()


if __name__ == "__main__":
    main()
