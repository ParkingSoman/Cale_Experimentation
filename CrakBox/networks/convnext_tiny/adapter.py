"""ConvNeXt-Tiny for the mapper: the positive control. A node is a neuron of the last block: c17:123. Its inputs are the 768 stream directions at that block, with names like c17s:45.
A stream direction has no card and no reading, so the control runs pole tests and the critic, and no parent test. It is the first test of the network interface."""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mapper.network import Network

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import net as cn  # noqa: E402

BATCH = 32
CURVES = os.path.join(cn.CLEAN, "curves")


class ConvNeXtTiny(Network):
    name = "convnext_tiny"

    def __init__(self, threads=4):
        self.threads = threads
        self._w = None
        self._photos = {}
        self._similar = {}
        self._premean = None

    @property
    def w(self):
        if self._w is None:
            torch.set_num_threads(self.threads)
            self._w = cn.load_weights()
        return self._w

    def _p(self, split):
        if split not in self._photos:
            self._photos[split] = cn.Photos(split)
        return self._photos[split]

    def _arr(self, split, key):
        return np.load(os.path.join(cn.STORE, f"{split}_{key}.npy"), mmap_mode="r")

    def layers(self):
        return ["c17"]

    def nodes(self, layer):
        return [f"c17:{j}" for j in range(3072)]

    def layer_of(self, node):
        return node.split(":")[0]

    def poles(self, node):
        return ("pos", "neg")

    def _j(self, node):
        return int(node.split(":")[1])

    def peaks(self, node, pole, split):
        j = self._j(node)
        if pole == "pos":
            return np.asarray(self._arr(split, "x")[:, j], dtype=np.float64), np.asarray(self._arr(split, "p")[:, j]).astype(np.int64)
        return -np.asarray(self._arr(split, "n")[:, j], dtype=np.float64), np.asarray(self._arr(split, "q")[:, j]).astype(np.int64)

    def _fields(self, split, examples):
        """Pre and neuron values for these examples."""
        P = self._p(split)
        pres, acts = [], []
        with torch.no_grad():
            for start in range(0, len(examples), BATCH):
                x = cn.Photos.tensor(P.raw(examples[start:start + BATCH]))
                pre, a = cn.block_parts(self.w, cn.upto_last(self.w, x), 3, cn.LAST)
                pres.append(pre.flatten(1, 2))
                acts.append(a.flatten(1, 2))
        return torch.cat(pres), torch.cat(acts)

    def value_at(self, node, pole, split, examples, positions):
        examples, positions = np.asarray(examples), np.asarray(positions)
        _, a = self._fields(split, examples)
        v = a[torch.arange(len(examples)), torch.as_tensor(positions), self._j(node)].numpy().astype(np.float64)
        return np.maximum(v, 0) if pole == "pos" else np.maximum(-v, 0)

    def _pre_mean(self):
        if self._premean is None:
            P = self._p("read")
            rng = np.random.default_rng(0)
            sel = np.sort(rng.choice(len(P.ix), 640, replace=False))
            pre, _ = self._fields("read", sel)
            self._premean = pre.mean((0, 1))
        return self._premean

    def inputs(self, node, split, examples, positions):
        examples, positions = np.asarray(examples), np.asarray(positions)
        j = self._j(node)
        pre, a = self._fields(split, examples)
        at = torch.arange(len(examples))
        p = f"stages.3.blocks.{cn.LAST}.mlp."
        D = (pre[at, torch.as_tensor(positions)] - self._pre_mean()) * self.w[p + "fc1.weight"][j]
        base = float(self._pre_mean() @ self.w[p + "fc1.weight"][j] + self.w[p + "fc1.bias"][j])
        true = a[at, torch.as_tensor(positions), j].numpy().astype(np.float64)
        return D.numpy().astype(np.float64), [f"c17s:{k}" for k in range(768)], np.full(len(examples), base), true

    def readers(self, node, split, count):
        """The classes it pushes, through its write to the stream and the head's weights. Norms taken as fixed, so this is a guide and not an exact sum."""
        j = self._j(node)
        p = f"stages.3.blocks.{cn.LAST}."
        write = self.w[p + "mlp.fc2.weight"][:, j] * self.w[p + "gamma"]
        W = self.w["head.fc.weight"] @ (write * self.w["head.norm.weight"]) if "head.norm.weight" in self.w else self.w["head.fc.weight"] @ write
        W = W - W.mean()
        return [(f"logit:{int(i)}", float(W[i])) for i in torch.argsort(W.abs(), descending=True)[:count]]

    def render(self, split, node, examples, positions, size):
        raw = self._p(split).raw(examples)
        return [Image.fromarray(raw[i]).resize((size, size)) for i in range(len(examples))]

    def similar(self, node, split, count):
        key = (node, split, count)
        if key not in self._similar:
            j = self._j(node)
            hm = np.asarray(self._arr(split, "m"), dtype=np.float32)
            u = hm[:, j] - hm[:, j].mean()
            corr = (hm - hm.mean(0)).T @ u / (hm.std(0) * u.std() * len(u) + 1e-9)
            corr[j] = -2
            self._similar[key] = [f"c17:{int(k)}" for k in np.argsort(-corr)[:count]]
        return self._similar[key]

    def output_effect(self, node, split, examples):
        j = self._j(node)
        P = self._p(split)
        total, count = 0.0, 0
        mean = float(np.asarray(self._arr("read", "m")[:, j], dtype=np.float64).mean())
        with torch.no_grad():
            for start in range(0, len(examples), BATCH):
                x = cn.Photos.tensor(P.raw(np.asarray(examples)[start:start + BATCH]))
                h = cn.upto_last(self.w, x)
                _, a = cn.block_parts(self.w, h, 3, cn.LAST)
                y = cn.head(self.w, cn.after_block(self.w, h, a, 3, cn.LAST))
                b = a.clone()
                b[..., j] = mean
                z = cn.head(self.w, cn.after_block(self.w, h, b, 3, cn.LAST))
                lp = F.log_softmax(y, 1)
                total += float((lp.exp() * (lp - F.log_softmax(z, 1))).sum())
                count += len(x)
        return total / max(count, 1)

    def evidence(self, tool, args):
        allowed = {"deep", "edit", "probe", "inputs", "readers", "classes"}
        if tool not in allowed:
            raise ValueError(f"no evidence tool called {tool}. Tools: {sorted(allowed)}")
        return [os.path.join(CURVES, ".venv", "bin", "python"), os.path.join(HERE, "ev.py"), tool, *args], CURVES

    def python(self):
        return os.path.join(CURVES, ".venv", "bin", "python")

    def evidence_help(self):
        return "deep NODE (prints a PNG path, then open it with Read), inputs NODE, readers NODE, edit NODE, probe NODE IMAGE..., classes WORD"

    def n_positions(self, node):
        return cn.GRID ** 2

    def example_labels(self, split, examples):
        return np.asarray(self._arr(split, "labels"))[np.asarray(examples)].tolist()

    def largest_in_layer(self, layer, split):
        return np.asarray(self._arr(split, "x").max(0), dtype=np.float64).tolist()

    def leak_patterns(self):
        return [r"\bc17s?\b", r"\b(unit|channel|node|neuron|filter|logit|layer|block|stage|direction)\s*#?\s*\d+\b", r"\b\d{1,4}\s*:\s*\d{1,4}\b"]
