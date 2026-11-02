"""Cache intact logits (fp16, 50k x 1008) and a one-to-one label map (Hungarian on train-split mean softmax)."""
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
from inception import *
m = load_model(); X, y = load_images(); tr, dv, se = split(y)
T = np.zeros((50000, 1008), np.float16)
with torch.no_grad():
    for i in range(0, 50000, 100):
        T[i:i+100] = from_3b(m, F.relu(pre_3b(m, to_3a(m, prep(X[i:i+100]))))).cpu().numpy()
np.save("cache/teacher_logits.npy", T)
P = np.zeros((1000, 1008)); Tt = torch.softmax(torch.from_numpy(T[tr].astype(np.float32)), 1).numpy()
np.add.at(P, y[tr], Tt)
r, c = linear_sum_assignment(-np.log(P + 1e-9)); lab = np.zeros(1000, int); lab[r] = c
np.save("label_map.npy", lab)
for nm, ix in [("train", tr), ("dev", dv), ("sealed(not used later)", [])]:
    if len(ix): print(nm, "top-1:", (T[ix][:, lab].argmax(1) == y[ix]).mean())
