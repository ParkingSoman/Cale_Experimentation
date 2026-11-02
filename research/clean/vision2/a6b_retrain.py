"""A6: collapse + brief retraining. Rules: VISION-LOG.md A6."""
import sys
sys.argv = ["t1_fold.py", "none"]
exec(open("a5b_late.py").read().split("res = {}")[0].split('"""', 2)[2])
import mlx.optimizers as optim
from mlx.utils import tree_flatten
ANA = np.where(rank < 20)[0]; rr = np.random.default_rng(7)
ORD["random"] = {b: list(np.random.default_rng(5).permutation(H[b])) for b in BLS}
TL = np.concatenate([np.array(m(mx.array(norm_img(np.asarray(X[ANA[i:i + 100]]))))) for i in range(0, len(ANA), 100)]); print("teacher logits", f"{time.time() - t0:.0f}s", flush=True)
F1 = [m.blocks[b].fc1 for b in range(NB)]; F2s = [m.blocks[b].fc2 for b in range(NB)]; HN, FC = m.head_norm, m.fc
def lin(Wt, bt):
    f = nn.Linear(Wt.shape[1], Wt.shape[0]); f.weight = mx.array(Wt.astype(np.float32)); f.bias = mx.array(bt.astype(np.float32)); return f
def late(x):
    for b in range(12, NB):
        if ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
        x = m.blocks[b](x)
    return m.head(x)
def top1():
    p = np.concatenate([np.array(m(mx.array(norm_img(np.asarray(X[EV[i:i + 100]]))))).argmax(1) for i in range(0, len(EV), 100)]); return p == yE
res = {}
TREE_DONE = {'before': 0.37366666666666665, 'after': 0.6373333333333333}
for name, O in [(k, ORD[k]) for k in ('energy', 'random')]:
    m = load_convnext()   # fresh model per arm (A6 bug: conv_dw/norm trained in one arm leaked into the next)
    for b in range(NB):   # fresh copies of all trainable layers
        m.blocks[b].fc1 = lin(np.array(F1[b].weight), np.array(F1[b].bias)); m.blocks[b].fc2 = lin(np.array(F2s[b].weight), np.array(F2s[b].bias))
    m.head_norm = nn.LayerNorm(HN.weight.shape[0], eps=1e-6); m.head_norm.weight, m.head_norm.bias = HN.weight, HN.bias; m.fc = lin(np.array(FC.weight), np.array(FC.bias))
    for b in O:
        rem = O[b][:int(.75 * H[b])]; W1 = np.array(m.blocks[b].fc1.weight); b1 = np.array(m.blocks[b].fc1.bias); W2 = np.array(m.blocks[b].fc2.weight); b2 = np.array(m.blocks[b].fc2.bias)
        b2 = b2 + W2[:, rem] @ np.array([mu[(b, n)] for n in rem])        # mean replacement
        W1[rem] = 0; b1[rem] = 0; W2[:, rem] = 0
        m.blocks[b].fc1 = lin(W1, b1); m.blocks[b].fc2 = lin(W2, b2)
    before = top1()
    m.freeze()
    for b in range(12, NB): m.blocks[b].unfreeze()
    m.head_norm.unfreeze(); m.fc.unfreeze()
    def loss_fn(model, xs, tl):
        lg = late(xs); lp = lg - mx.logsumexp(lg, axis=1, keepdims=True); tp = tl - mx.logsumexp(tl, axis=1, keepdims=True)
        return (mx.exp(tp) * (tp - lp)).sum(1).mean()
    opt = optim.Adam(learning_rate=1e-4); lvg = nn.value_and_grad(m, loss_fn); perm = rr.permutation(len(ANA)); ls = []
    for st in range(0, len(ANA), 50):
        ix = np.sort(perm[st:st + 50]); xs = mx.stop_gradient(m.stream(mx.array(norm_img(np.asarray(X[ANA[ix]]))), upto=12))
        l, g = lvg(m, xs, mx.array(TL[ix])); opt.update(m, g); mx.eval(m.parameters(), opt.state); ls.append(float(l))
    m.unfreeze(); after = top1()
    res[name] = {"before": float(before.mean()), "after": float(after.mean()), "loss_first_last": [float(np.mean(ls[:20])), float(np.mean(ls[-20:]))], "per_img_after": after.astype(int).tolist()}
    print(name, {k: v for k, v in res[name].items() if k != "per_img_after"}, f"{time.time() - t0:.0f}s", flush=True)
res["tree"] = TREE_DONE; print("tree (from first run)", TREE_DONE)
d = None
if False: d = np.array(res["tree"]["per_img_after"]) - np.array(res["energy"]["per_img_after"]); #bs = [d[rr.integers(0, len(d), len(d))].mean() for _ in range(2000)]
#res["tree_minus_energy_after"] = [float(d.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]; print(res["tree_minus_energy_after"])
json.dump(res, open(os.path.join(RES, "vision-a6b-retrain.json"), "w"))
