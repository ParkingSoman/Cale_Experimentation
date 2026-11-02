# serves: H-sym
"""Part C estimator for H-sym (COARSE-THEORY.md): orientation harmonics of the photo law, no network.

eps_sym(n) = |sum m^2 e^{i n psi}| / sum m^2 over pixels and images, n = 2, 4; gradient by Gaussian derivatives
(sd 1.5 px) on grayscale. Floor: the same estimator on white-noise images. Standard errors by bootstrap over images.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from inception import load_images, split

X, y = load_images()
tr, _, _ = split(y)
idx = tr[:2000]
rng = np.random.default_rng(0)


def harmonics(img):
    g = img.astype(np.float64)
    gx = gaussian_filter(g, 1.5, order=(0, 1))
    gy = gaussian_filter(g, 1.5, order=(1, 0))
    m2 = gx ** 2 + gy ** 2
    psi = np.arctan2(gy, gx)
    return m2.sum(), (m2 * np.exp(2j * psi)).sum(), (m2 * np.exp(4j * psi)).sum()


def run(images):
    H = np.array([harmonics(im) for im in images])          # (n, 3) complex
    def eps(rows):
        tot = H[rows, 0].real.sum()
        return np.abs(H[rows, 1].sum()) / tot, np.abs(H[rows, 2].sum()) / tot, np.angle(H[rows, 2].sum())
    e2, e4, ph4 = eps(np.arange(len(H)))
    boots = np.array([eps(rng.integers(0, len(H), len(H)))[:2] for _ in range(300)])
    return e2, e4, ph4, boots.std(0)


gray = (X[idx].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32))
e2, e4, ph4, se = run(gray)
noise = rng.normal(size=(2000, 224, 224)).astype(np.float32)
f2, f4, fph4, fse = run(noise)
print(f"photos (2000 train): eps_sym(2) = {e2:.4f} (se {se[0]:.4f})   eps_sym(4) = {e4:.4f} (se {se[1]:.4f})   "
      f"phase of n=4 harmonic {np.degrees(ph4):.0f} deg (0 = horizontal/vertical excess, 180 = diagonal excess)")
print(f"white-noise floor:   eps_sym(2) = {f2:.4f} (se {fse[0]:.4f})   eps_sym(4) = {f4:.4f} (se {fse[1]:.4f})")
