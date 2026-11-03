# serves: H-map
import numpy as np, os
from PIL import Image, ImageFilter
out = "/private/tmp/claude-501/east/"
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
g = np.zeros((224, 224, 3), np.uint8)
for y in range(224):
    g[y] = [20, 90 + y // 3, 170 + y // 5]
Image.fromarray(g).save(out + "blue_smooth.png")
Image.fromarray(np.full((224, 224, 3), 200, np.uint8)).save(out + "grey_flat.png")
r = np.zeros((224, 224, 3), np.uint8)
r[..., 0] = rng.integers(120, 255, (224, 224)); r[..., 1] = rng.integers(0, 60, (224, 224)); r[..., 2] = rng.integers(0, 60, (224, 224))
Image.fromarray(r).save(out + "red_noise.png")
Image.fromarray(np.full((224, 224, 3), [200, 30, 30], np.uint8)).save(out + "red_flat.png")
b = np.zeros((224, 224, 3), np.uint8)
b[..., 2] = rng.integers(120, 255, (224, 224)); b[..., 1] = rng.integers(0, 90, (224, 224))
Image.fromarray(b).save(out + "blue_noise.png")
