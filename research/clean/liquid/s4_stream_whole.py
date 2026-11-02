"""S4: stream width, whole model. Does one global rotation + slice keep the model usable at 90% and 75% width?

  python3 research/clean/liquid/s4_stream_whole.py --d 518,432

Exact emulation of the sliced model: the word table rows are projected onto the basis P (E -> E P P^T) and every
layer's reads and writes are restricted to P (s3_stream.sliced_layer). The stream then never leaves span(P). The output
head can stay tied: logits = (E P)(P^T diag(gamma_f) P)(P^T x_hat), one extra d' x d' matrix, so the head is unchanged
here. A true sliced model computes rms over d' coordinates instead of 576; that is a constant factor sqrt(576/d') that
folds into the norm scales, so the emulation is exact. Slicing to d' removes (1 - d'/576) of ALL weights, word table too.
Arms: bases from S3 (global only), with and without the per-layer LS read refit.
Controls in this harness: 9->6 query heads cut_ls in every layer, and MLP cut_ls_mix in every layer at the same
share of layer weights as the stream arm (each layer fitted on the intact model, as Z4 did).
"""
import argparse
from lib import *  # noqa: F403
from lib import _capture, _cv_ridge  # noqa: E402
from s3_stream import collect, bases, sliced_layer, eig_desc, with_saliency, READS_IN, READS_POST  # noqa: E402
from contract import load_method  # noqa: E402


def whole_slice(S, P, acc=None):
    edits = []
    for L in range(len(S.model.model.layers)):
        edits += sliced_layer(S, L, P, acc)
    emb = S.model.model.embed_tokens
    E = np.array(emb.weight, dtype=np.float64)
    new = nn.Embedding(E.shape[0], E.shape[1]); new.weight = mx.array((E @ P @ P.T).astype(np.float32))
    # tied head must keep the ORIGINAL table (see docstring): wrap so only the lookup uses the projected table
    class Emb(nn.Module):
        def __init__(self, lookup, orig):
            super().__init__(); self.lookup, self.orig = lookup, orig
        def __call__(self, ids):
            return self.lookup(ids)
        def as_linear(self, x):
            return self.orig.as_linear(x)
    edits.append((S.model.model, "embed_tokens", Emb(new, emb)))
    return edits


def block_basis(S, acc, layers, kind, sal):
    """Basis from a subset of layers: 'pca' (norm-input second moment) or 'read_act' (activity-weighted reads)."""
    if kind == "pca":
        return with_saliency(eig_desc(sum(acc[("x", l)] + acc[("h", l)] for l in layers)), sal)
    M = np.zeros((D_MODEL, D_MODEL))
    for l in layers:
        lay = S.model.model.layers[l]
        for names, gam, par, key in ((READS_IN, lay.input_layernorm.weight, lay.self_attn, "x"), (READS_POST, lay.post_attention_layernorm.weight, lay.mlp, "h")):
            g = np.array(gam, dtype=np.float64); C = acc[(key, l)] / np.trace(acc[(key, l)])
            for n in names:
                W = np.array(getattr(par, n).weight, dtype=np.float64) * g
                e = np.einsum("ij,jk,ik->i", W, C, W); Gm = (W * e[:, None]).T @ W; M += Gm / np.trace(Gm)
    return with_saliency(eig_desc(M), sal)


class Entry(nn.Module):
    """Projects the stream onto the block's subspace before the block (emulates the d' x d' joining matrix)."""
    def __init__(self, blk, P):
        super().__init__(); self.blk = blk; self.PP = mx.array((P @ P.T).astype(np.float32))
        self.use_sliding = getattr(blk, "use_sliding", False)
    def __call__(self, x, mask=None, cache=None):
        return self.blk(x @ self.PP, mask, cache)


def block_slice(S, acc, dp, kind, nblk, sal, ls):
    """One basis per block of layers. Stream re-projected at each block entry (stored cost: d' x d' per boundary)."""
    layers = S.model.model.layers; nl = len(layers); size = -(-nl // nblk); edits = []; Ps = []
    for b in range(nblk):
        ls_ = list(range(b * size, min(nl, (b + 1) * size)))
        P = block_basis(S, acc, ls_, kind, sal)[:, :dp]; Ps.append(P)
        for L in ls_:
            edits += sliced_layer(S, L, P, acc if ls else None)
    for b in range(1, nblk):
        L0 = b * size
        if L0 < nl:
            # sliced_layer edits sub-modules of the block; the Entry wraps the block object itself
            edits.append((layers, L0, Entry(layers[L0], Ps[b])))
    emb = S.model.model.embed_tokens; E = np.array(emb.weight, dtype=np.float64)
    new = nn.Embedding(E.shape[0], E.shape[1]); new.weight = mx.array((E @ Ps[0] @ Ps[0].T).astype(np.float32))
    edits.append((S.model.model, "embed_tokens", Emb(new, emb)))
    return edits


class Emb(nn.Module):
    def __init__(self, lookup, orig):
        super().__init__(); self.lookup, self.orig = lookup, orig
    def __call__(self, ids):
        return self.lookup(ids)
    def as_linear(self, x):
        return self.orig.as_linear(x)


def heads_cut_ls_all(S, r=1):
    edits = []
    for L, lay in enumerate(S.model.model.layers):
        at = lay.self_attn; Wo = np.array(at.o_proj.weight, dtype=np.float64)
        z = _capture(S.model, S.cal[::2], {"z": (at, "o_proj")})["z"]
        Ya = (z.astype(np.float64) @ Wo.T).astype(np.float32)
        en = [np.sum((z[:, h * 64:(h + 1) * 64].astype(np.float64) @ Wo[:, h * 64:(h + 1) * 64].T) ** 2) for h in range(N_HEADS)]
        kept = sorted(h for g in range(N_KV) for h in sorted(range(3 * g, 3 * g + 3), key=lambda h: en[h])[r:])
        kc = np.concatenate([np.arange(h * 64, (h + 1) * 64) for h in kept])
        W = np.zeros_like(Wo); W[:, kc] = _cv_ridge(z[:, kc], Ya).T
        edits.append((at, "o_proj", SinkGuard(Lin(W), Lin(Wo))))
    return edits


def mlp_cut_all(S, K):
    mod = load_method(os.path.join(CLEAN, "coarse", "methods", "cut_ls_mix.py"))
    return [(lay, "mlp", mod.build(S.model, S.tok, L, K, S.cal[:64])) for L, lay in enumerate(S.model.model.layers)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--d", default="518,432"); ap.add_argument("--bases", default="pca_global,read_glob,read_act,coord")
    ap.add_argument("--blocks", default="6"); ap.add_argument("--kinds", default="pca,read_act")
    ap.add_argument("--controls", type=int, default=1)
    a = ap.parse_args(); S = Setup(); t = Timer(); acc = collect(S); print(f"stats {t()}s", flush=True)
    B, sal = bases(S, acc, 16); print("saliency", sal, flush=True); rows = []
    def rec(name, dp, frac, edits):
        res = S.score(edits)
        row = {"step": "S4", "arm": name, "d": dp, "bytes_frac_layers": round(frac, 4), **{k: boot(v) for k, v in res.items()}}
        rows.append(row); print(f"  {name:22s} d{dp} frac {frac:.3f}  prose {row['prose'][0]:+.4f}  code {row['code'][0]:+.4f}  ({t()}s)", flush=True)
    for dp in map(int, a.d.split(",")):
        frac = 1 - dp / D_MODEL
        for k in a.bases.split(","):
            rec(k, dp, frac, whole_slice(S, B[k][:, :dp]))
            if k != "coord":
                rec(k + "_ls", dp, frac, whole_slice(S, B[k][:, :dp], acc))
        for nb in map(int, filter(None, a.blocks.split(","))):
            for kind in a.kinds.split(","):
                for ls in (False, True):
                    rec(f"blocks{nb}_{kind}" + ("_ls" if ls else ""), dp, frac, block_slice(S, acc, dp, kind, nb, sal, ls))
        if a.controls:
            K = D_FF - int(round(frac * LAYER_PARAMS / NEURON_PARAMS))
            rec(f"mlp_cut_ls_mix_K{K}", dp, frac, mlp_cut_all(S, K))
        print("saved", save("s4-stream-whole", rows)); rows.clear()
    if a.controls:
        rec("heads_9to6_cut_ls", None, 3 * HEAD_PARAMS / LAYER_PARAMS, heads_cut_ls_all(S, 1))
        print("saved", save("s4-stream-whole", rows))


if __name__ == "__main__":
    main()
