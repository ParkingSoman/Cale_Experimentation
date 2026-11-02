"""S2b: flexible vs fixed reference for the head-size dial (standing check, research/clean/FLEXIBLE-VS-FIXED.md).

  python3 research/clean/liquid/s2b_fixed_ref.py --layers 16 --keep 0.5 --secs 60

At one layer and one keep fraction (both head sides), compare at the same storage:
  flexible   qk_act pair drop + vo_svd (one nested order, no refit)
  ls         + CV-ridge o_proj refit (per size)
  fixed      start from ls, then train the kept q/k/v/o weights for --secs seconds with Adam on the whole-model KL to the
             original next-word distribution (calibration blocks, block position 0 excluded). Dropped rows stay zero.
"""
import argparse
import mlx.optimizers as optim
from lib import *  # noqa: F403
from lib import _capture  # noqa: E402
from s2_headsize import vo_bases, vo_apply, qk_drop, pair_norm, refit_o, AttnGuard, new_attn, lin, G  # noqa: E402


def run(S, L, keep, secs, rows):
    lay = S.model.model.layers[L]; at = lay.self_attn; t = Timer()
    Wq, Wk, Wv, Wo = (np.array(getattr(at, n).weight, dtype=np.float64) for n in ("q_proj", "k_proj", "v_proj", "o_proj"))
    cap = _capture(S.model, S.cal, {"z": (at, "o_proj"), "xa": (at, "q_proj")})
    z = cap["z"]; xa = cap["xa"][::4].astype(np.float64); Ya = (z.astype(np.float64) @ Wo.T).astype(np.float32)
    qn, kn = pair_norm(xa @ Wq.T, N_HEADS), pair_norm(xa @ Wk.T, N_KV)
    s_act = np.array([sum(qn[h] for h in range(g * G, (g + 1) * G)) * kn[g] for g in range(N_KV)])
    r = int(round(64 * keep)); bf = 2 * (64 - r) * N_KV * 2304 / LAYER_PARAMS
    Wq2, Wk2 = qk_drop(Wq, Wk, s_act, r // 2); Wv2, Wo2 = vo_apply(Wv, Wo, vo_bases(z, Wo), r)
    Wo_ls = refit_o(S, at, [("q_proj", lin(Wq2)), ("k_proj", lin(Wk2)), ("v_proj", lin(Wv2))], Ya, r)
    arms = {"flexible": new_attn(S, at, q_proj=Wq2, k_proj=Wk2, v_proj=Wv2, o_proj=Wo2),
            "ls": new_attn(S, at, q_proj=Wq2, k_proj=Wk2, v_proj=Wv2, o_proj=Wo_ls)}
    # fixed reference: train from ls
    a = new_attn(S, at, q_proj=Wq2, k_proj=Wk2, v_proj=Wv2, o_proj=Wo_ls)
    masks = {n: mx.array((np.abs(w) > 0).astype(np.float32)) for n, w in (("q_proj", Wq2), ("k_proj", Wk2), ("v_proj", Wv2), ("o_proj", Wo_ls))}
    for n in ("q_proj", "k_proj", "v_proj", "o_proj"):  # o mask: kept columns only
        if n == "o_proj":
            m = np.zeros_like(Wo_ls); kc = np.concatenate([np.arange(h * 64, h * 64 + r) for h in range(N_HEADS)]); m[:, kc] = 1
            masks[n] = mx.array(m.astype(np.float32))
    S.model.freeze(); a.unfreeze(); a.freeze(keys=["rope"], strict=False)
    guard = AttnGuard(a, at)
    opt = optim.Adam(learning_rate=1e-4)
    def loss_fn(x, p_ref):
        lay.self_attn = guard
        lg = S.model(x); lay.self_attn = at
        lp = lg - mx.logsumexp(lg, -1, keepdims=True)
        return (mx.exp(p_ref) * (p_ref - lp)).sum(-1)[:, 1:].mean()
    vg = nn.value_and_grad(a, loss_fn)
    rng = np.random.default_rng(0); n = S.cal.shape[0]; step = 0; t0 = time.time(); losses = []
    while time.time() - t0 < secs:
        idx = mx.array(rng.choice(n, 4, replace=False)); x = S.cal[idx][:, :-1]
        ref = S.model(x); p_ref = ref - mx.logsumexp(ref, -1, keepdims=True)
        loss, gr = vg(x, mx.stop_gradient(p_ref)); opt.update(a, gr)
        for nm, m in masks.items():
            getattr(a, nm).weight = getattr(a, nm).weight * m
        mx.eval(a.parameters(), opt.state); losses.append(float(loss)); step += 1
    arms["fixed"] = a
    print(f"L{L} keep{keep} fixed: {step} steps, KL {np.mean(losses[:10]):.4f} -> {np.mean(losses[-10:]):.4f}  ({t()}s)", flush=True)
    res = {}
    for name, m in arms.items():
        res[name] = S.score([(lay, "self_attn", AttnGuard(m, at))])
        row = {"step": "S2b", "layer": L, "keep": keep, "arm": name, "bytes_frac": round(bf, 4), "secs": secs if name == "fixed" else 0,
               **{k: boot(v) for k, v in res[name].items()}}
        rows.append(row); print(f"  L{L} keep{keep} {name:9s} prose {row['prose'][0]:+.4f}  code {row['code'][0]:+.4f}", flush=True)
    d = {k: boot(res["flexible"][k] - res["fixed"][k]) for k in res["fixed"]}
    rows.append({"step": "S2b", "layer": L, "keep": keep, "diff": "flexible - fixed", **d})
    print(f"  flexible - fixed: prose {np.round(d['prose'], 4).tolist()} code {np.round(d['code'], 4).tolist()}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="16"); ap.add_argument("--keep", default="0.5"); ap.add_argument("--secs", type=float, default=60)
    a = ap.parse_args(); S = Setup(); rows = []
    for L in map(int, a.layers.split(",")):
        for k in map(float, a.keep.split(",")):
            run(S, L, k, a.secs, rows)
    print("saved", save("s2b-fixed", rows))


if __name__ == "__main__":
    main()
