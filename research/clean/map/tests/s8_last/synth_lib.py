"""Helpers for synthetic tests: draw images, get max z of several neurons."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_lib import maps
from PIL import Image, ImageDraw


def sky(top=(150, 190, 235), bot=(215, 230, 245)):
    a = np.zeros((224, 224, 3), np.float32)
    for i in range(224): a[i] = np.array(top) + (np.array(bot) - np.array(top)) * i / 223
    return Image.fromarray(a.astype(np.uint8))


def run(imgs, targets):
    """imgs: dict name -> PIL image. Returns dict name -> {target: (max, argmax rc)}."""
    names = list(imgs); arr = np.stack([np.asarray(imgs[k].convert("RGB").resize((224, 224))) for k in names])
    Z = maps(arr, targets); out = {}
    for i, k in enumerate(names):
        out[k] = {t: float(Z[t][i].max()) for t in targets}
    return out, Z, names


def table(out, targets, ref):
    """print rows: name, then value / ref per target."""
    print("stimulus".ljust(22) + "".join(f"b{t[0]}n{t[1]}".rjust(12) for t in targets))
    for k, v in out.items():
        print(k.ljust(22) + "".join(f"{v[t]:7.2f}({v[t]/ref[t]:+.2f})" for t in targets))
