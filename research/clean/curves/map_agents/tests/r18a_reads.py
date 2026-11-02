# serves: H-map
# Prediction (could fail): for a channel, mean activation on photos whose label is among its 20 strongest-positive reader
# classes is >= 2x the mean elsewhere, and the channel's write on those classes is a large share of its write energy.
import numpy as np
m = np.load('cache/map/resnet18_meta.npz')
W, lab = m['W'].astype(float), m['labels']
H = np.load('cache/map/resnet18_h.npy', mmap_mode='r')
Wc = W - W.mean(0, keepdims=True)
print('unit  lift(mean h on top20 reader classes / elsewhere)  share of write energy on those 20 classes  frac of total h>3 photos inside')
for u in [147, 153, 161, 179, 221, 327, 366, 371, 416]:
    top = np.argsort(-Wc[:, u])[:20]
    inn = np.isin(lab, top)
    h = np.asarray(H[:, u], float)
    e = Wc[:, u] ** 2
    big = h > 3
    print(u, round(h[inn].mean() / h[~inn].mean(), 2), round(e[top].sum() / e.sum(), 2), round(inn[big].mean(), 2), 'base', round(inn.mean(), 3))
