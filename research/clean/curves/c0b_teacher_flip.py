"""Cache intact logits of mirrored images (fp16, 50k x 1008)."""
import numpy as np, torch
from inception import *
m = load_model(); X, y = load_images()
T = np.zeros((50000, 1008), np.float16)
with torch.no_grad():
    for i in range(0, 50000, 100):
        T[i:i+100] = from_3b(m, F.relu(pre_3b(m, to_3a(m, prep(np.asarray(X[i:i+100])[:, :, ::-1]))))).cpu().numpy()
np.save("cache/teacher_logits_flip.npy", T); print("ok")
