"""A small fake network for tests. It needs no torch and no photos. The truth is known, so the judge's scoring can be checked exactly.
The last ten nodes of "bot" read the raw input and have no negative side, so that the tests reach those paths too. A spike node has one huge firing, so that a sheet must refuse it."""
import numpy as np
from PIL import Image

from mapper.network import Network


class FakeNet(Network):
    name = "fake"

    def __init__(self, width=30, examples=2000, seed=0):
        self.width = width
        rng = np.random.default_rng(seed)
        self.pos_values = {s: rng.gamma(1.0, 1.0, (examples, width)) for s in ("read", "sealed")}
        self.neg_values = {s: rng.gamma(1.0, 1.0, (examples, width)) for s in ("read", "sealed")}
        self.n_inputs = 6
        self.weights = rng.normal(size=(width, self.n_inputs))
        for s in ("read", "sealed"):
            self.pos_values[s][:, 3] = rng.gamma(1.0, .01, examples)
            self.pos_values[s][0, 3] = 100.0  # top:3 and raw:3 are spike nodes

    def layers(self):
        return ["top", "bot", "raw"]

    def nodes(self, layer):
        return [f"{layer}:{c}" for c in range(self.width)]

    def layer_of(self, node):
        return node.split(":")[0]

    def poles(self, node):
        return ("pos",) if node.startswith("raw") else ("pos", "neg")

    def _c(self, node):
        return int(node.split(":")[1])

    def peaks(self, node, pole, split):
        v = (self.pos_values if pole == "pos" else self.neg_values)[split][:, self._c(node)]
        return v.copy(), np.zeros(len(v), dtype=np.int64)

    def value_at(self, node, pole, split, examples, positions):
        return self.peaks(node, pole, split)[0][np.asarray(examples)]

    def has_inputs(self, node):
        return not node.startswith("raw")

    def n_positions(self, node):
        return 1

    def inputs(self, node, split, examples, positions):
        if node.startswith("raw"):
            raise ValueError("this node reads the raw input")
        examples = np.asarray(examples)
        c = self._c(node)
        v = self.pos_values[split][examples, c] - self.neg_values[split][examples, c]
        below = "bot" if node.startswith("top") else "raw"
        w = np.abs(self.weights[c]) / np.abs(self.weights[c]).sum()
        D = np.outer(v, w)
        names = [f"{below}:{j}" for j in range(self.n_inputs)] + [f"{node} (skip)"]
        D = np.hstack([D, np.zeros((len(examples), 1))])
        base = np.zeros(len(examples))
        return D, names, base, base + D.sum(1)

    def input_kept(self, name):
        return name.endswith(" (skip)")

    def readers(self, node, split, count):
        if node.startswith("top"):
            return [(f"logit:{j}", float(self.weights[j, 0])) for j in range(count)]
        above = "top" if node.startswith("bot") else "bot"
        return [(f"{above}:{j}", float(self.weights[j, 0])) for j in range(count)]

    def words_of(self, name):
        return f"output number {name.split(':')[1]}" if name.startswith("logit:") else None

    def output_names(self):
        return [f"logit:{j}" for j in range(self.width)]

    def leak_patterns(self):
        return [r"\b(unit|channel)\s*\d+\b"]

    def canonical(self, name):
        return name.replace(" (skip)", "")

    def output_scores(self, examples, split):
        return {n: float(self.pos_values[split][np.asarray(examples), self._c(n)].mean()) for n in self.nodes("top")}

    def battery(self):
        return [(f"item {i}", Image.new("RGB", (32, 32), (i * 20, 90, 90))) for i in range(12)]

    def battery_values(self, node):
        return (np.arange(12) * (self._c(node) + 1) % 7) / 7.0

    def render(self, split, node, examples, positions, size):
        return [Image.new("RGB", (size, size), (int(i) % 255, 100, 100)) for i in examples]

    def similar(self, node, split, count):
        c = self._c(node)
        return [f"{self.layer_of(node)}:{(c + k) % self.width}" for k in range(1, count + 1)]

    def output_effect(self, node, split, examples):
        return 0.01

    def evidence(self, tool, args):
        return ["echo", tool, *args], "."
