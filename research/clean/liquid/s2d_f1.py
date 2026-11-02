"""S2d: which head side misses the gains that the fixed reference found (S2b)? A 15 s local fit per side.

  python3 research/clean/liquid/s2d_f1.py --layers 12,16,22 --keep 0.5 --secs 15

Start: the per-size LS state (qk_act pairs + vo_svd + o refit). Then Adam on the attention block's own output error
(cached block inputs, position 0 excluded), training only the kept weights of one side:
  f1_qk    q and k (kept pairs)          f1_vo  v and o (kept dims)          f1_all  all four
Zero extra storage. Minutes reported. Dev scored as in S2.
"""
import argparse
import mlx.optimizers as optim
from lib import *  # noqa: F403
from lib import _capture  # noqa: E402
from s2_headsize import vo_bases, vo_apply, qk_drop, pair_norm, refit_o, AttnGuard, new_attn, lin, G  # noqa: E402


class GrabIn(nn.Module):
    def __init__(self, inner):
        super().__init__(); self.inner, self.x = inner, []
    def __call__(self, x, mask=None, cache=None):
        self.x.append(x); return self.inner(x, mask, cache)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="12,16,22"); ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--secs", type=float, default=15); ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args(); S = Setup(); rows = []
    for L in map(int, a.layers.split(",")):
        lay = S.model.model.layers[L]; at = lay.self_attn; t = Timer()
        Wq, Wk, Wv, Wo = (np.array(getattr(at, n).weight, dtype=np.float64) for n in ("q_proj", "k_proj", "v_proj", "o_proj"))
        cap = _capture(S.model, S.cal, {"z": (at, "o_proj"), "xa": (at, "q_proj")})
        z = cap["z"]; xa = cap["xa"][::4].astype(np.float64); Ya = (z.astype(np.float64) @ Wo.T).astype(np.float32)
        qn, kn = pair_norm(xa @ Wq.T, N_HEADS), pair_norm(xa @ Wk.T, N_KV)
        s_act = np.array([sum(qn[h] for h in range(g * G, (g + 1) * G)) * kn[g] for g in range(N_KV)])
        r = int(round(64 * a.keep)); bf = 2 * (64 - r) * N_KV * 2304 / LAYER_PARAMS
        Wq2, Wk2 = qk_drop(Wq, Wk, s_act, r // 2); Wv2, Wo2 = vo_apply(Wv, Wo, vo_bases(z, Wo), r)
        Wo_ls = refit_o(S, at, [("q_proj", lin(Wq2)), ("k_proj", lin(Wk2)), ("v_proj", lin(Wv2))], Ya, r)
        del cap, z
        # cache normalized block inputs (input of self_attn) and original attention outputs for 32 cal blocks
        g = GrabIn(at); lay.self_attn = g; blocks = mx.concatenate([S.cal[:16], S.cal[-16:]], 0)
        for i in range(0, 32, 4):
            mx.eval(S.model(blocks[i:i + 4, :-1]))
        lay.self_attn = at; Xin = mx.concatenate(g.x, 0); Yref = mx.concatenate([at(Xin[i:i + 4], "causal") for i in range(0, 32, 4)], 0)
        mx.eval(Xin, Yref)
        kc = np.concatenate([np.arange(h * 64, h * 64 + r) for h in range(N_HEADS)])
        mo = np.zeros_like(Wo); mo[:, kc] = 1
        masks = {"q_proj": (np.abs(Wq2) > 0), "k_proj": (np.abs(Wk2) > 0), "v_proj": (np.abs(Wv2) > 0), "o_proj": mo > 0}
        res = {"ls": S.score([(lay, "self_attn", AttnGuard(new_attn(S, at, q_proj=Wq2, k_proj=Wk2, v_proj=Wv2, o_proj=Wo_ls), at))])}
        for side, names in (("f1_qk", ("q_proj", "k_proj")), ("f1_vo", ("v_proj", "o_proj")), ("f1_all", ("q_proj", "k_proj", "v_proj", "o_proj"))):
            m = new_attn(S, at, q_proj=Wq2, k_proj=Wk2, v_proj=Wv2, o_proj=Wo_ls)
            m.freeze()
            for n in names:  # unfreeze(keys=...) matches local parameter names ("weight"), not submodules
                getattr(m, n).unfreeze()
            mk = {n: mx.array(masks[n].astype(np.float32)) for n in names}
            opt = optim.Adam(learning_rate=a.lr)
            def loss_fn(x, y):
                return ((m(x, "causal") - y)[:, 1:] ** 2).mean()
            vg = nn.value_and_grad(m, loss_fn); rng = np.random.default_rng(0); t0 = time.time(); steps = 0; ls_ = []
            while time.time() - t0 < a.secs:
                idx = mx.array(rng.choice(32, 4, replace=False))
                loss, gr = vg(Xin[idx], Yref[idx]); opt.update(m, gr)
                for n, k in mk.items():
                    getattr(m, n).weight = getattr(m, n).weight * k
                mx.eval(m.parameters(), opt.state); ls_.append(float(loss)); steps += 1
            print(f"L{L} {side}: {steps} steps, mse {np.mean(ls_[:5]):.5f} -> {np.mean(ls_[-5:]):.5f}", flush=True)
            res[side] = S.score([(lay, "self_attn", AttnGuard(m, at))])
        for nm, v in res.items():
            row = {"step": "S2d", "layer": L, "keep": a.keep, "arm": nm, "bytes_frac": round(bf, 4), "secs": a.secs if nm != "ls" else 0,
                   **{k: boot(x) for k, x in v.items()}}
            rows.append(row); print(f"  L{L} {nm:7s} prose {row['prose'][0]:+.4f}  code {row['code'][0]:+.4f}", flush=True)
        for x in ("f1_qk", "f1_vo", "f1_all"):
            d = {k: boot(res[x][k] - res["ls"][k]) for k in res[x]}
            rows.append({"step": "S2d", "layer": L, "keep": a.keep, "diff": f"{x} - ls", **d})
            print(f"  L{L} {x} - ls: prose {np.round(d['prose'], 4).tolist()} code {np.round(d['code'], 4).tolist()}", flush=True)
        print(f"L{L} {t()}s", flush=True)
    print("saved", save("s2d-f1", rows))


if __name__ == "__main__":
    main()
