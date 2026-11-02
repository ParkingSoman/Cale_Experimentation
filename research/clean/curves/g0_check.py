"""Gate 0: split path == full model; speed; empirical label map (train split); top-1 on dev."""
import time, json, numpy as np, torch
from inception import *
m = load_model(); X, y = load_images(); tr, dv, se = split(y)
with torch.no_grad():
    x = prep(X[:8]); full = m.softmax2_pre_activation_matmul if False else None
    a3 = to_3a(m, x); h = F.relu(pre_3b(m, a3)); lg = from_3b(m, h)
    p_full = m(x); p_mine = torch.softmax(lg, 1)
    print("max |diff| softmax:", (p_full - p_mine).abs().max().item())
    t = time.time(); n = 0
    P = np.zeros((1000, 1008), np.float32); cnt = np.zeros(1000)
    idx = tr[::7]  # 5k train images for the label map
    for i in range(0, len(idx), 100):
        b = idx[i:i+100]; lg = from_3b(m, F.relu(pre_3b(m, to_3a(m, prep(X[b])))))
        pr = torch.softmax(lg, 1).cpu().numpy(); np.add.at(P, y[b], pr); n += len(b)
    torch.mps.synchronize(); print(f"{n/(time.time()-t):.0f} img/s full forward")
    lab = P.argmax(1); print("unique mapped outputs:", len(set(lab)))
    np.save("label_map.npy", lab)
    corr = 0
    for i in range(0, len(dv), 100):
        b = dv[i:i+100]; lg = from_3b(m, F.relu(pre_3b(m, to_3a(m, prep(X[b]))))).cpu().numpy()
        corr += (lg[:, lab].argmax(1) == y[b]).sum()
    print("dev top-1 (1000-way over mapped outputs):", corr / len(dv))
