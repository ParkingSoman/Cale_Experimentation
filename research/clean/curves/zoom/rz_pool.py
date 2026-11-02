# serves: H-map
"""Pooled last-layer activity h_c(x) = mean over the 49 positions of relu(z_c(x, s)) for the ResNet-18 cache, plus the
head (fc weight W, bias b) and the 1000 class names. Writes cache/r18/h_<split>.npy (float32) and head.npz."""
import os
import numpy as np
import torchvision

C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "r18")
for s in ("train", "sealed"):
    Z = np.load(os.path.join(C, f"z_{s}.npy"), mmap_mode="r")
    H = np.zeros((len(Z), 512), np.float32)
    for b in range(0, len(Z), 2000):
        H[b:b + 2000] = np.maximum(Z[b:b + 2000].astype(np.float32), 0).mean(2)
    np.save(os.path.join(C, f"h_{s}.npy"), H)
wts = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
sd = wts.get_state_dict(progress=False)
np.savez(os.path.join(C, "head.npz"), W=sd["fc.weight"].detach().numpy(), b=sd["fc.bias"].detach().numpy(),
         names=np.array(wts.meta["categories"]))
H = np.load(os.path.join(C, "h_train.npy")); hd = np.load(os.path.join(C, "head.npz"))
Y = np.load(os.path.join(C, "y_train.npy"))
print("max |W h + b - y|", float(np.abs(H @ hd["W"].T + hd["b"] - Y).max()))
