"""ConvNeXt (timm layout) in MLX, loaded from timm safetensors. Vision-first lab (memory feedback-vision-first-good-models).
Block: x + gamma * fc2(gelu(fc1(ln(dwconv7(x))))). Neurons = fc1 units (4x width)."""
import io, os, json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from safetensors.numpy import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
HF = os.path.join(ROOT, "research", "clean", "hf")
RES = os.path.join(ROOT, "results")
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


class Block(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.conv_dw = nn.Conv2d(C, C, 7, padding=3, groups=C); self.norm = nn.LayerNorm(C, eps=1e-6)
        self.fc1 = nn.Linear(C, 4 * C); self.fc2 = nn.Linear(4 * C, C); self.gamma = mx.ones((C,))

    def pre(self, x):            # input of fc1
        return self.norm(self.conv_dw(x))

    def __call__(self, x):
        return x + self.gamma * self.fc2(nn.gelu(self.fc1(self.pre(x))))


class ConvNeXt(nn.Module):
    def __init__(self, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768)):
        super().__init__()
        self.stem = nn.Conv2d(3, dims[0], 4, stride=4); self.stem_norm = nn.LayerNorm(dims[0], eps=1e-6)
        self.down_norm = [nn.LayerNorm(dims[i - 1], eps=1e-6) for i in range(1, 4)]
        self.down = [nn.Conv2d(dims[i - 1], dims[i], 2, stride=2) for i in range(1, 4)]
        self.blocks = []; self.stage_of = []
        for s, (d, c) in enumerate(zip(depths, dims)):
            for _ in range(d):
                self.blocks.append(Block(c)); self.stage_of.append(s)
        self.head_norm = nn.LayerNorm(dims[-1], eps=1e-6); self.fc = nn.Linear(dims[-1], 1000)

    def stream(self, x, upto=None):
        """Residual stream entering block `upto` (None = after the last block)."""
        x = self.stem_norm(self.stem(x)); prev = 0
        for i, b in enumerate(self.blocks):
            s = self.stage_of[i]
            if s != prev:
                x = self.down[s - 1](self.down_norm[s - 1](x)); prev = s
            if upto is not None and i == upto:
                return x
            x = b(x)
        return x

    def head(self, x):
        return self.fc(self.head_norm(x.mean((1, 2))))

    def run_from(self, x, start):
        """Continue from the stream entering block `start` to logits."""
        prev = self.stage_of[start]
        for i in range(start, len(self.blocks)):
            s = self.stage_of[i]
            if s != prev:
                x = self.down[s - 1](self.down_norm[s - 1](x)); prev = s
            x = self.blocks[i](x)
        return self.head(x)

    def __call__(self, x):
        return self.head(self.stream(x))


def conv_w(w):   # torch (out, in, kh, kw) -> mlx (out, kh, kw, in)
    return mx.array(np.transpose(w, (0, 2, 3, 1)))


def load_convnext(name="tiny", depths=(3, 3, 9, 3), dims=(96, 192, 384, 768)):
    w = load_file(os.path.join(HF, "convnext", name, "model.safetensors"))
    m = ConvNeXt(depths, dims)
    m.stem.weight, m.stem.bias = conv_w(w["stem.0.weight"]), mx.array(w["stem.0.bias"])
    m.stem_norm.weight, m.stem_norm.bias = mx.array(w["stem.1.weight"]), mx.array(w["stem.1.bias"])
    for i in range(1, 4):
        m.down_norm[i - 1].weight, m.down_norm[i - 1].bias = mx.array(w[f"stages.{i}.downsample.0.weight"]), mx.array(w[f"stages.{i}.downsample.0.bias"])
        m.down[i - 1].weight, m.down[i - 1].bias = conv_w(w[f"stages.{i}.downsample.1.weight"]), mx.array(w[f"stages.{i}.downsample.1.bias"])
    k = 0
    for s, d in enumerate(depths):
        for j in range(d):
            p = f"stages.{s}.blocks.{j}."; b = m.blocks[k]; k += 1
            b.conv_dw.weight, b.conv_dw.bias = conv_w(w[p + "conv_dw.weight"]), mx.array(w[p + "conv_dw.bias"])
            b.norm.weight, b.norm.bias = mx.array(w[p + "norm.weight"]), mx.array(w[p + "norm.bias"])
            sq = lambda a: a.reshape(a.shape[:2])     # atto/nano store the MLP as 1x1 convs
            b.fc1.weight, b.fc1.bias = mx.array(sq(w[p + "mlp.fc1.weight"])), mx.array(w[p + "mlp.fc1.bias"])
            b.fc2.weight, b.fc2.bias = mx.array(sq(w[p + "mlp.fc2.weight"])), mx.array(w[p + "mlp.fc2.bias"])
            b.gamma = mx.array(w[p + "gamma"])
    m.head_norm.weight, m.head_norm.bias = mx.array(w["head.norm.weight"]), mx.array(w["head.norm.bias"])
    m.fc.weight, m.fc.bias = mx.array(w["head.fc.weight"]), mx.array(w["head.fc.bias"])
    mx.eval(m.parameters()); return m


SHARDS = ["val-00000-of-00002-b5248be478d25e41.parquet", "val-00001-of-00002-85f3d9c8fa1edb63.parquet"]


def load_images(size=224):
    """All 50k ImageNet val images (resized to 256 short side by the mirror), center crop 224, uint8, memory-mapped."""
    xf, yf = os.path.join(HF, "imagenet", "val_224.npy"), os.path.join(HF, "imagenet", "val_labels.npy")
    if not os.path.exists(yf):
        import pyarrow.parquet as pq
        from PIL import Image
        X = np.lib.format.open_memmap(xf, mode="w+", dtype=np.uint8, shape=(50000, size, size, 3)); ys = []; k = 0
        for sh in SHARDS:
            t = pq.read_table(os.path.join(HF, "imagenet", "data", sh)).to_pandas()
            for d in t["image"]:
                im = Image.open(io.BytesIO(d["bytes"])).convert("RGB"); W, H = im.size; s = 256 / min(W, H)
                if abs(s - 1) > 1e-3: im = im.resize((round(W * s), round(H * s)), Image.BICUBIC)
                W, H = im.size; l, tp = (W - size) // 2, (H - size) // 2
                X[k] = np.array(im.crop((l, tp, l + size, tp + size))); k += 1
            ys.append(t["label"].values.astype(np.int32))
        X.flush(); np.save(yf, np.concatenate(ys)); assert k == 50000
    return np.load(xf, mmap_mode="r"), np.load(yf)


def split(y):
    """Per class: first 20 images = analysis, the other 30 = sealed."""
    rank = np.zeros(len(y), int)
    for c in range(1000):
        ix = np.where(y == c)[0]; rank[ix] = np.arange(len(ix))
    return np.where(rank < 20)[0], np.where(rank >= 20)[0]


def norm_img(X):
    return (X.astype(np.float32) / 255.0 - MEAN) / STD


def class_names():
    import re
    txt = open(os.path.join(HF, "imagenet", "README.md")).read()
    sec = txt.split("names:")[1].split("splits:")[0]
    return [re.sub(r"^\s*'?\d+'?:\s*", "", l).strip().strip("'\"") for l in sec.strip().split("\n") if re.match(r"^\s*'?\d+'?:", l)]


def logits(m, X, bs=100):
    return np.concatenate([np.array(m(mx.array(norm_img(X[i:i + bs])))) for i in range(0, len(X), bs)])
