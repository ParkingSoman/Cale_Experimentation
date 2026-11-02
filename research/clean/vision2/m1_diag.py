"""M1 diagnosis: are the position groups meaningful? variance of child z explained by groups; dominant coords in pre."""
import numpy as np, mlx.core as mx
from sklearn.cluster import MiniBatchKMeans
from convnext import *
m = load_convnext(); X, y = load_images(); ana, _ = split(y)
idx = np.concatenate([ana[y[ana] == c][:2] for c in range(1000)])
b = 17; PRE, Z, IMG = [], [], []
for i in range(0, len(idx), 50):
    x = m.stream(mx.array(norm_img(np.asarray(X[idx[i:i + 50]]))), upto=b); pre = m.blocks[b].pre(x); z = m.blocks[b].fc1(pre)
    PRE.append(np.array(pre.reshape(-1, pre.shape[-1]))); Z.append(np.array(z.reshape(-1, z.shape[-1]))); IMG.append(np.repeat(np.arange(i, i + len(pre)), 49))
PRE, Z, IMG = np.concatenate(PRE), np.concatenate(Z), np.concatenate(IMG)
v = PRE.var(0); o = np.argsort(-v); print("pre variance share of top 5 coords:", np.round(v[o[:5]] / v.sum(), 3), "coords", o[:5])
print("pre mean |x| top coords:", np.round(np.abs(PRE).mean(0)[np.argsort(-np.abs(PRE).mean(0))[:5]], 2))
def r2(g, K):
    cnt = np.bincount(g, minlength=K)[:, None]; mu = np.zeros((K, Z.shape[1])); np.add.at(mu, g, Z); mu /= np.maximum(cnt, 1)
    return 1 - ((Z - mu[g]) ** 2).sum(0) / ((Z - Z.mean(0)) ** 2).sum(0)
for name, data in (("pre (euclid)", PRE), ("pre, unit norm", PRE / np.linalg.norm(PRE, axis=1, keepdims=True)),
                   ("pre, top-2 var coords removed", np.delete(PRE, o[:2], 1))):
    g = MiniBatchKMeans(2000, batch_size=4096, n_init=1, random_state=0).fit_predict(data); r = r2(g, 2000)
    print(f"{name}: child z R^2 from groups median {np.median(r):.2f}; top-1% positions per neuron: share of groups that hold them, median", end=" ")
    top = np.argsort(-Z, 0)[:int(.01 * len(Z))]; print(np.median([len(np.unique(g[top[:, j]])) for j in range(0, 3072, 30)]))
g = IMG; print("image as group (49 positions each): child z R^2 median", np.round(np.median(r2(g, len(idx))), 2))
