# serves: H-map
# Prediction (could fail): the channels that carry N's decision for wheeled-cycle classes form a district whose own read
# (the cycle logit-direction read of the whole layer) is a gate: outside it, the district's write costs little.
import json
import numpy as np
m = np.load('cache/map/resnet18_meta.npz')
W, lab = m['W'].astype(float), m['labels']
H = np.load('cache/map/resnet18_h.npy').astype(float)
C = [444, 870, 880, 671]
d = (W[C] - W.mean(0)).mean(0)               # cycle direction, one weight per channel
inn = np.isin(lab, C)
lift = H[inn].mean(0) - H[~inn].mean(0)
sc = d * lift                                 # how much each channel pushes cycles up on cycle photos
o = np.argsort(-sc)[:16]
print('top channels', o.tolist(), np.round(sc[o], 2).tolist())
print('179 rank', int(np.where(np.argsort(-sc) == 179)[0][0]), '327 rank', int(np.where(np.argsort(-sc) == 327)[0][0]))
k = 12
U = o[:k].tolist()
read = H @ d                                  # the layer's cycle read
for q in (0.90, 0.97):
    th = float(np.quantile(read, q))
    print(q, 'gate theta', th, 'cycle photos inside', float((read[inn] > th).mean()))
    claims = [dict(name=f'cycle district k={k}, layer cycle-read > q{q}', units=U,
                   region=dict(gate=dict(units=list(range(512)), weights=d.tolist(), theta=th)))]
    json.dump(claims, open(f'map_agents/claims/resnet18_r18a_cycle_{int(q*100)}.json', 'w'))
