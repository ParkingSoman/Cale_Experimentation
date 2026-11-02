"""S5: one ranked list at one layer. At the same weights removed, does mixing head size + MLP neurons beat MLP alone?

  python3 research/clean/liquid/s5_mix.py --layers 12,16,22 --budgets 0.10,0.25,0.40

For each budget B (share of the layer's weights), try every allocation (qk keep, vo keep) on a small grid; the rest
of B comes from MLP neurons (cut_ls on the mixed calibration, fitted on the ORIGINAL MLP inputs so it can be cached).
Head size uses the flexible nested order (qk_act pairs + vo_svd, no refit).
The allocation is CHOSEN on a separate selection set (prose train blocks 200-223 + code chars 1.6M+), never on dev.
Reported on dev: chosen mix vs MLP-only cut_ls (same MLP method, isolates the value of mixing) and vs MLP-only rcf_v1
(the strong single-knob reference).
"""
import argparse, itertools
from lib import *  # noqa: F403
from lib import _capture  # noqa: E402
from lib import code_blocks  # noqa: E402
from s2_headsize import vo_bases, vo_apply, qk_drop, pair_norm, AttnGuard, new_attn, G  # noqa: E402
from contract import mlp_io, cut_ls, load_method  # noqa: E402
from tree_lib import hidden  # noqa: E402

QK, VO = (1.0, 0.75, 0.5), (1.0, 0.75, 0.5, 0.25)


class MLPCache:
    """cut_ls for any K from one capture: neuron order and LS on the original inputs."""

    def __init__(self, S, L):
        X, _ = mlp_io(S.model, S.cal, L); self.X = X[::2]
        Gm, Um, Dm = orig_weights(S.model, L); self.G, self.U, self.D = Gm, Um, Dm
        h = hidden(self.X, Gm, Um); self.Y = h @ Dm.T
        self.order = np.argsort(-(np.linalg.norm(h, axis=0) * np.linalg.norm(Dm, axis=0))); self.h = h

    def layer(self, K):
        if K >= D_FF:
            return SwiGLULayer(self.G, self.U, self.D)
        idx = np.sort(self.order[:K]); hk = self.h[:, idx]
        Dk = np.linalg.solve(hk.T @ hk + 1e-6 * np.trace(hk.T @ hk) / K * np.eye(K), hk.T @ self.Y).T
        return SwiGLULayer(self.G[idx], self.U[idx], Dk)


def head_edit(S, L, qk, vo, pre):
    lay = S.model.model.layers[L]; at = lay.self_attn
    if qk == 1.0 and vo == 1.0:
        return []
    Wq, Wk, Wv, Wo, s_act, bases = pre
    Wq2, Wk2 = qk_drop(Wq, Wk, s_act, int(round(64 * qk)) // 2) if qk < 1 else (Wq, Wk)
    Wv2, Wo2 = vo_apply(Wv, Wo, bases, int(round(64 * vo))) if vo < 1 else (Wv, Wo)
    return [(lay, "self_attn", AttnGuard(new_attn(S, at, q_proj=Wq2, k_proj=Wk2, v_proj=Wv2, o_proj=Wo2), at))]


def run_layer(S, sel, L, budgets, rows):
    lay = S.model.model.layers[L]; at = lay.self_attn; t = Timer()
    Wq, Wk, Wv, Wo = (np.array(getattr(at, n).weight, dtype=np.float64) for n in ("q_proj", "k_proj", "v_proj", "o_proj"))
    cap = _capture(S.model, S.cal, {"z": (at, "o_proj"), "xa": (at, "q_proj")})
    xa = cap["xa"][::4].astype(np.float64); qn, kn = pair_norm(xa @ Wq.T, N_HEADS), pair_norm(xa @ Wk.T, N_KV)
    s_act = np.array([sum(qn[h] for h in range(g * G, (g + 1) * G)) * kn[g] for g in range(N_KV)])
    pre = (Wq, Wk, Wv, Wo, s_act, vo_bases(cap["z"], Wo)); del cap
    mc = MLPCache(S, L); print(f"L{L} prep {t()}s", flush=True)
    rcf = load_method(os.path.join(CLEAN, "coarse", "methods", "rcf_v1.py"))
    for B in budgets:
        cands = []
        for qk, vo in itertools.product(QK, VO):
            head_frac = ((1 - qk) + (1 - vo)) * 64 * N_KV * 2304 / LAYER_PARAMS
            n_rm = int(round((B - head_frac) * LAYER_PARAMS / NEURON_PARAMS))
            if n_rm < 0:
                continue
            edits = head_edit(S, L, qk, vo, pre) + [(lay, "mlp", mc.layer(D_FF - n_rm))]
            old = S.dev; S.dev = sel; old_base = S.base; S.base = S.sel_base
            d = S.score(edits); S.dev = old; S.base = old_base
            cands.append((float(d["prose"].mean() + d["code"].mean()), qk, vo, n_rm))
        cands.sort()
        for c in cands:
            rows.append({"step": "S5", "layer": L, "budget": B, "select_score": c[0], "qk": c[1], "vo": c[2], "mlp_removed": c[3]})
        best = cands[0]; mlp_only = [c for c in cands if c[1] == 1.0 and c[2] == 1.0][0]
        print(f"L{L} B{B}: best qk{best[1]} vo{best[2]} mlp-{best[3]} sel {best[0]:.4f}; mlp-only sel {mlp_only[0]:.4f}  ({t()}s)", flush=True)
        arms = {"mix": head_edit(S, L, best[1], best[2], pre) + [(lay, "mlp", mc.layer(D_FF - best[3]))],
                "mlp_only_cut": [(lay, "mlp", mc.layer(D_FF - mlp_only[3]))]}
        layer = rcf.build(S.model, S.tok, L, D_FF - mlp_only[3], S.cal[:64], log=lambda *a: None)
        arms["mlp_only_rcf"] = [(lay, "mlp", layer)] + list(layer.extra)
        res = {k: S.score(v) for k, v in arms.items()}
        for k, v in res.items():
            row = {"step": "S5", "layer": L, "budget": B, "arm": k, "qk": best[1] if k == "mix" else 1.0, "vo": best[2] if k == "mix" else 1.0,
                   **{tt: boot(x) for tt, x in v.items()}}
            rows.append(row); print(f"  L{L} B{B} {k:13s} prose {row['prose'][0]:+.4f}  code {row['code'][0]:+.4f}", flush=True)
        for a in ("mlp_only_cut", "mlp_only_rcf"):
            d = {k: boot(res["mix"][k] - res[a][k]) for k in res["mix"]}
            rows.append({"step": "S5", "layer": L, "budget": B, "diff": f"mix - {a}", **d})
            print(f"  L{L} B{B} mix - {a}: prose {np.round(d['prose'], 4).tolist()} code {np.round(d['code'], 4).tolist()}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="12,16,22"); ap.add_argument("--budgets", default="0.10,0.25,0.40")
    a = ap.parse_args(); S = Setup()
    sel = {"prose": token_blocks(S.tok, "train", 224)[200:224], "code": code_blocks(S.tok, 24, 1600000)}
    S.sel_base = {k: block_ce(S.model, v) for k, v in sel.items()}; rows = []
    for L in map(int, a.layers.split(",")):
        run_layer(S, sel, L, list(map(float, a.budgets.split(","))), rows)
        print("saved", save("s5-mix", rows)); rows.clear()


if __name__ == "__main__":
    main()
