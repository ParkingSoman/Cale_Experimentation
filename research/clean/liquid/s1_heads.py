"""S1: query heads at one layer. Can removed heads be captured by existing routes better than cut + refit?

  python3 research/clean/liquid/s1_heads.py --layers 12,16,22 --r 1,2

r = heads removed per key/value group (1: 9->6 heads, 2: 9->3 heads). Removal is emulated by zeroing the removed
heads' o_proj columns; bytes are counted as removed (q rows + o columns). Arms (all with the position-0 sink guard):
  cut        zero the weakest heads (write energy), no refit
  cut_ls     + CV-ridge refit of the kept heads' o_proj on the original attention output   (fair control)
  merge_sum  removed head merged into its most similar kept head: q rows averaged, o columns summed
  merge_ls   merge_sum's q, then CV-ridge refit of o_proj
  head_rcf   CV-ridge fold of [kept head outputs, layer L-1 MLP hiddens] -> original attention output, added into the
             kept heads' o_proj and layer L-1's down_proj (zero extra storage)
Equal-bytes MLP references: r query heads per group = 3r heads = 128r neurons. cut_ls_mix and rcf_v1 at K=1536-128r.
"""
import argparse
from lib import *  # noqa: F403
from lib import _capture, _cv_ridge  # noqa: E402
from contract import load_method  # noqa: E402

G = 3  # query heads per key/value group


def heads_of(z, h):
    return z[:, h * HEAD_DIM:(h + 1) * HEAD_DIM]


def cols(heads):
    return np.concatenate([np.arange(h * HEAD_DIM, (h + 1) * HEAD_DIM) for h in heads])


def o_module(Wo_new, Wo):
    return SinkGuard(Lin(Wo_new), Lin(Wo))


def capture(S, L):
    lay = S.model.model.layers[L]
    mods = {"z": (lay.self_attn, "o_proj"), "xp": (S.model.model.layers[L - 1], "mlp")}
    return _capture(S.model, S.cal, mods)


def run_layer(S, L, rs, rows):
    lay = S.model.model.layers[L]; at = lay.self_attn; t = Timer()
    Wo = np.array(at.o_proj.weight, dtype=np.float64); Wq = np.array(at.q_proj.weight, dtype=np.float64)
    cap = capture(S, L); z = cap["z"]
    Ya = (z.astype(np.float64) @ Wo.T).astype(np.float32)
    energy = np.array([np.sum((heads_of(z, h).astype(np.float64) @ Wo[:, h * HEAD_DIM:(h + 1) * HEAD_DIM].T) ** 2) for h in range(N_HEADS)])
    W_heads = [(heads_of(z, h).astype(np.float64) @ Wo[:, h * HEAD_DIM:(h + 1) * HEAD_DIM].T)[::8].ravel() for h in range(N_HEADS)]
    Gp, Up, Dp = orig_weights(S.model, L - 1); Xp = cap["xp"].astype(np.float64)
    Hp = (silu(Xp @ Gp.T) * (Xp @ Up.T)).astype(np.float32); del Xp
    print(f"L{L} capture {t()}s  head energy share {np.round(energy / energy.sum(), 3).tolist()}", flush=True)

    for r in rs:
        removed, kept, partner = [], [], {}
        for g in range(N_KV):
            hs = list(range(g * G, (g + 1) * G)); order = sorted(hs, key=lambda h: energy[h])
            rem, kp = order[:r], order[r:]; removed += rem; kept += kp
            for h in rem:  # most similar kept head (cosine of writes)
                partner[h] = max(kp, key=lambda k: np.dot(W_heads[h], W_heads[k]) / np.linalg.norm(W_heads[h]) / np.linalg.norm(W_heads[k]))
        kept = sorted(kept); kc = cols(kept); bytes_frac = len(removed) * HEAD_PARAMS / LAYER_PARAMS
        arms = {}
        Wc = Wo.copy(); Wc[:, cols(removed)] = 0
        arms["cut"] = [(at, "o_proj", o_module(Wc, Wo))]
        Wls = np.zeros_like(Wo); Wls[:, kc] = _cv_ridge(z[:, kc], Ya).T
        arms["cut_ls"] = [(at, "o_proj", o_module(Wls, Wo))]
        # merge: q rows averaged over each kept head and the removed heads assigned to it; o columns summed
        Wq_m = Wq.copy(); Wo_m = Wo.copy(); Wo_m[:, cols(removed)] = 0
        for k in kept:
            grp = [k] + [h for h in removed if partner[h] == k]
            Wq_m[cols([k])] = np.mean([Wq[cols([h])] for h in grp], 0)
            Wo_m[:, cols([k])] = np.sum([Wo[:, cols([h])] for h in grp], 0)
        q_m = nn.Linear(D_MODEL, N_HEADS * HEAD_DIM, bias=False); q_m.weight = mx.array(Wq_m.astype(np.float32))
        arms["merge_sum"] = [(at, "q_proj", q_m), (at, "o_proj", o_module(Wo_m, Wo))]
        old_q = at.q_proj; at.q_proj = q_m
        zm = _capture(S.model, S.cal, {"z": (at, "o_proj")})["z"]; at.q_proj = old_q
        Wml = np.zeros_like(Wo); Wml[:, kc] = _cv_ridge(zm[:, kc], Ya).T; del zm
        arms["merge_ls"] = [(at, "q_proj", q_m), (at, "o_proj", o_module(Wml, Wo))]
        Wf = _cv_ridge(np.concatenate([z[:, kc], Hp], 1), Ya)
        Wr = np.zeros_like(Wo); Wr[:, kc] = Wf[:len(kc)].T; Cp = Wf[len(kc):]
        arms["head_rcf"] = [(at, "o_proj", o_module(Wr, Wo)),
                            (S.model.model.layers[L - 1], "mlp", SinkGuard(SwiGLULayer(Gp, Up, Dp + Cp.T), SwiGLULayer(Gp, Up, Dp)))]
        print(f"L{L} r{r} removed {sorted(removed)} fits {t()}s", flush=True)
        # equal-bytes MLP references
        K = D_FF - 128 * r
        for name in ("cut_ls_mix", "rcf_v1"):
            mod = load_method(os.path.join(CLEAN, "coarse", "methods", f"{name}.py"))
            layer = mod.build(S.model, S.tok, L, K, S.cal[:64], log=lambda *a: None) if name == "rcf_v1" else mod.build(S.model, S.tok, L, K, S.cal[:64])
            arms[f"mlp_{name}"] = [(lay, "mlp", layer)] + list(getattr(layer, "extra", []))
        res = {}
        for name, edits in arms.items():
            res[name] = S.score(edits)
            row = {"step": "S1", "layer": L, "r": r, "arm": name, "bytes_frac": round(bytes_frac, 4),
                   **{k: boot(v) for k, v in res[name].items()}}
            rows.append(row)
            print(f"  L{L} r{r} {name:16s} prose {row['prose'][0]:+.4f}  code {row['code'][0]:+.4f}", flush=True)
        for a, b in (("head_rcf", "cut_ls"), ("merge_ls", "cut_ls"), ("head_rcf", "mlp_rcf_v1"), ("cut_ls", "mlp_cut_ls_mix")):
            d = {k: boot(res[a][k] - res[b][k]) for k in res[a]}
            rows.append({"step": "S1", "layer": L, "r": r, "diff": f"{a} - {b}", **d})
            print(f"  L{L} r{r} {a} - {b}: prose {np.round(d['prose'], 4).tolist()} code {np.round(d['code'], 4).tolist()}", flush=True)
        print(f"L{L} r{r} scored {t()}s", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="12,16,22"); ap.add_argument("--r", default="1,2")
    a = ap.parse_args(); S = Setup(); rows = []
    for L in map(int, a.layers.split(",")):
        run_layer(S, L, list(map(int, a.r.split(","))), rows)
        print("saved", save("s1-heads", rows)); rows.clear()


if __name__ == "__main__":
    main()
