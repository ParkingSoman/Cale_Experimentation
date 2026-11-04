"""ResNet-18 for the mapper. It wraps ev2.py and net.py from the curves track, and adds nothing to the network.
A node is a channel at one layer: s4b1:179 (stream), s4b1i:179 (inner), s0:12 (the stem). Examples are photos. A position is an index into the layer's grid."""
import functools
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mapper.network import Network

CURVES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "research", "clean", "curves"))
TOOLS = os.path.join(CURVES, "map_agents", "resnet18")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
import ev2  # noqa: E402
import net as rn  # noqa: E402

BATCH = 32
# ev2 recomputes a layer's mean over all reading photos on each call. Stored once here, since the means never change.
ev2.mean_of = functools.lru_cache(maxsize=None)(ev2.mean_of)


class ResNet18(Network):
    name = "resnet18"

    def __init__(self, threads=4):
        self.threads = threads
        self._m = None
        self._photos = {}
        self._similar = {}

    @property
    def m(self):
        if self._m is None:
            torch.set_num_threads(self.threads)
            self._m = rn.load()
        return self._m

    def layers(self):
        out = []
        for s in (4, 3, 2, 1):
            out += [f"s{s}b1", f"s{s}b1i", f"s{s}b0", f"s{s}b0i"]
        return out + ["s0"]

    def nodes(self, layer):
        s = int(layer[1])
        return [f"{layer}:{c}" for c in range(rn.WIDTH[s])]

    def layer_of(self, node):
        return node.split(":")[0]

    def poles(self, node):
        return ("pos",) if node.startswith("s0") else ("pos", "neg")

    def _p(self, split):
        if split not in self._photos:
            self._photos[split] = rn.Photos(split)
        return self._photos[split]

    def _arr(self, split, key):
        return np.load(os.path.join(rn.STORE, split, key + ".npy"), mmap_mode="r")

    def peaks(self, node, pole, split):
        n = ev2.parse(node)
        pre = ev2.pre(n)
        if pole == "pos":
            k = "i" if n["inner"] else "h"
            v, p = pre + k + "x", pre + k + "p"
        else:
            if n["s"] == 0:
                raise ValueError("the stem stores no negative pole")
            v, p = (pre + "an", pre + "ap") if n["inner"] else (pre + "zn", pre + "zp")
        values = np.asarray(self._arr(split, v)[:, n["c"]], dtype=np.float64)
        return (values if pole == "pos" else -values), np.asarray(self._arr(split, p)[:, n["c"]]).astype(np.int64)

    def _field_at(self, node, split, examples, positions):
        n = ev2.parse(node)
        out = []
        P = self._p(split)
        for start in range(0, len(examples), BATCH):
            ex, pos = examples[start:start + BATCH], positions[start:start + BATCH]
            f = ev2.value_field(self.m, P.tensor(ex), n).flatten(1)
            out.append(f[torch.arange(len(ex)), torch.as_tensor(pos)].numpy())
        return np.concatenate(out).astype(np.float64)

    def value_at(self, node, pole, split, examples, positions):
        v = self._field_at(node, split, np.asarray(examples), np.asarray(positions))
        return np.maximum(v, 0) if pole == "pos" else np.maximum(-v, 0)

    def inputs(self, node, split, examples, positions):
        n = ev2.parse(node)
        if n["s"] == 0:
            raise ValueError("the stem reads pixels, so it has no input nodes")
        P = self._p(split)
        Ds, bases, trues, names = [], [], [], None
        examples, positions = np.asarray(examples), np.asarray(positions)
        for start in range(0, len(examples), BATCH):
            ex, pos = examples[start:start + BATCH], positions[start:start + BATCH]
            D, names, base, true = ev2.exact_inputs(self.m, P.tensor(ex), n, pos)
            Ds.append(D)
            bases.append(base)
            trues.append(true)
        return np.concatenate(Ds), names, np.concatenate(bases), np.concatenate(trues)

    def readers(self, node, split, count):
        """The same arithmetic as ev2's readers view, returned as data. Top-layer readers are the logits, named logit:CLASS."""
        n = ev2.parse(node)
        s, b, c = n["s"], n["b"], n["c"]
        v, p = self.peaks(node, "pos", split)
        ex = np.sort(np.argsort(-v)[:32])
        x = self._p(split).tensor(ex)
        with torch.no_grad():
            _, o = rn.forward(self.m, x)
            if s == 4 and b == 1 and not n["inner"]:
                W = self.m.fc.weight[:, c] - self.m.fc.weight[:, c].mean()
                order = torch.argsort(W.abs(), descending=True)[:count]
                return [(f"logit:{int(i)}", float(W[i])) for i in order]
            blocks = rn.stages(self.m)
            if n["inner"]:
                bl = blocks[s][b]
                val, mean, conv, bn, nm = o[(s, b)]["inner"], ev2.mean_of(ev2.pre(n) + "im"), bl.conv2, bl.bn2, (lambda j: f"s{s}b{b}:{j}")
            else:
                ts, tb = (s, 1) if (s > 0 and b == 0) else (s + 1, 0)
                bl = blocks[ts][tb]
                val, mean, conv, bn = o[(s, b)]["h"], ev2.mean_of(f"s{s}b{b}_hm"), bl.conv1, bl.bn1
                nm = lambda j, ts=ts, tb=tb: f"s{ts}b{tb}i:{j}"  # noqa: E731
            dev = (val[:, c] - mean[c])[:, None]
            k = conv.kernel_size[0]
            w = conv.weight[:, c][:, None]
            out = F.conv2d(dev, w.view(-1, 1, k, k), stride=conv.stride, padding=conv.padding) * ev2.scale(bn).view(1, -1, 1, 1)
            flat = out.flatten(2)
            signed = flat.gather(2, flat.abs().argmax(2, keepdim=True))[..., 0]
        me = signed.mean(0).numpy()
        return [(nm(int(j)), float(me[j])) for j in np.argsort(-np.abs(me))[:count]]

    def render(self, split, node, examples, positions, size):
        n = ev2.parse(node)
        s = n["s"]
        raw = self._p(split).raw(examples)
        g, half = ev2.grid(s), ev2.CROP[s] // 2
        out = []
        for j in range(len(examples)):
            r, c = int(positions[j]) // g, int(positions[j]) % g
            cy, cx = int((r + .5) * rn.STRIDE[s]), int((c + .5) * rn.STRIDE[s])
            r0, c0 = min(max(cy - half, 0), 224 - 2 * half), min(max(cx - half, 0), 224 - 2 * half)
            out.append(Image.fromarray(raw[j][r0:r0 + 2 * half, c0:c0 + 2 * half]).resize((size, size)))
        return out

    def similar(self, node, split, count):
        key = (node, split, count)
        if key not in self._similar:
            n = ev2.parse(node)
            hm = np.asarray(self._arr(split, ev2.pre(n) + ("im" if n["inner"] else "hm")), dtype=np.float64)
            u = hm[:, n["c"]] - hm[:, n["c"]].mean()
            corr = (hm - hm.mean(0)).T @ u / (hm.std(0) * u.std() * len(u) + 1e-9)
            corr[n["c"]] = -2
            self._similar[key] = [f"{ev2.name(n, int(j))}" for j in np.argsort(-corr)[:count]]
        return self._similar[key]

    def _forward_fill(self, x, n):
        """The whole network, with this one node carrying its mean at every position."""
        m, s, b, c = self.m, n["s"], n["b"], n["c"]

        def fill(h, key):
            h = h.clone()
            h[:, c] = ev2.mean_of(key)[c].to(h)
            return h
        h = rn.stem(m, x)
        if s == 0:
            h = fill(h, "s0b0_hm")
        for ts, lay in rn.stages(m).items():
            for tb, bl in enumerate(lay):
                inner = torch.relu(bl.bn1(bl.conv1(h)))
                if (ts, tb) == (s, b) and n["inner"]:
                    inner = fill(inner, f"s{ts}b{tb}_im")
                skip = h if bl.downsample is None else bl.downsample(h)
                out = torch.relu(bl.bn2(bl.conv2(inner)) + skip)
                if (ts, tb) == (s, b) and not n["inner"]:
                    out = fill(out, f"s{ts}b{tb}_hm")
                h = out
        return rn.head(m, h)

    def weak_firing_kl(self, nodes, frac, examples, split):
        """KL on the outputs when firings under this share of each node's largest are set to 0, on the scope's nodes only. The check behind the firing limit."""
        by_place = {}
        for node in nodes:
            n = ev2.parse(node)
            v, _ = self.peaks(node, "pos", "read")
            by_place.setdefault((n["s"], n["b"], n["inner"]), []).append((n["c"], float(v.max())))

        def cut(h, place):
            if place not in by_place:
                return h
            h = h.clone()
            for c, big in by_place[place]:
                h[:, c] = torch.where(h[:, c] < frac * big, torch.zeros_like(h[:, c]), h[:, c])
            return h

        m = self.m
        P = self._p(split)
        total, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(examples), BATCH):
                x = P.tensor(np.asarray(examples)[start:start + BATCH])
                y = m(x)
                h = cut(rn.stem(m, x), (0, 0, False))
                for ts, lay in rn.stages(m).items():
                    for tb, bl in enumerate(lay):
                        inner = cut(torch.relu(bl.bn1(bl.conv1(h))), (ts, tb, True))
                        skip = h if bl.downsample is None else bl.downsample(h)
                        h = cut(torch.relu(bl.bn2(bl.conv2(inner)) + skip), (ts, tb, False))
                z = rn.head(m, h)
                lp = F.log_softmax(y, 1)
                total += float((lp.exp() * (lp - F.log_softmax(z, 1))).sum())
                count += len(x)
        return total / max(count, 1)

    def weak_firing_by_layer(self, nodes, fracs, examples, split):
        out = {}
        for layer in self.layers():
            here = [n for n in nodes if self.layer_of(n) == layer]
            if here:
                out[layer] = {str(f): round(self.weak_firing_kl(here, f, examples, split), 5) for f in fracs}
        return out

    def output_effect(self, node, split, examples):
        n = ev2.parse(node)
        P = self._p(split)
        total, count = 0.0, 0
        examples = np.asarray(examples)
        with torch.no_grad():
            for start in range(0, len(examples), BATCH):
                x = P.tensor(examples[start:start + BATCH])
                y = self.m(x)
                z = self._forward_fill(x, n)
                lp = F.log_softmax(y, 1)
                total += float((lp.exp() * (lp - F.log_softmax(z, 1))).sum(1).sum())
                count += len(x)
        return total / max(count, 1)

    def evidence(self, tool, args):
        if tool == "near":
            here = os.path.dirname(os.path.abspath(__file__))
            return [os.path.join(CURVES, ".venv", "bin", "python"), "-W", "ignore", os.path.join(here, "near_view.py"), *args, os.path.join(os.path.dirname(os.path.dirname(here)), "runs", "drawn", "near")], CURVES
        allowed = {"deep", "edit", "photo", "probe", "fv", "inputs", "readers", "bundles", "trace", "acts", "classes", "photos-of"}
        if tool not in allowed:
            raise ValueError(f"no evidence tool called {tool}. Tools: {sorted(allowed)}")
        if tool == "photo":
            args = [*args, "--dir", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runs", "drawn", "photos")]
        return [os.path.join(CURVES, ".venv", "bin", "python"), os.path.join(TOOLS, "ev2.py"), tool, *args], CURVES

    def largest_in_layer(self, layer, split):
        n = ev2.parse(self.nodes(layer)[0])
        key = ev2.pre(n) + ("ix" if n["inner"] else "hx")
        return np.asarray(self._arr(split, key).max(0), dtype=np.float64).tolist()

    def example_labels(self, split, examples):
        return np.asarray(self._arr(split, "labels"))[np.asarray(examples)].tolist()

    def judged_poles(self, node):
        """The negative pole's test failed its control here twice, and it passed on ConvNeXt-Tiny (LOG). So the lead is this network's regime: a ReLU cuts the negative side off."""
        return ("pos",)

    def canonical(self, name):
        return name.replace(" (skip)", "")

    def goal_examples(self, goal, split):
        labels = np.asarray(self._arr(split, "labels"))
        return np.where(np.isin(labels, np.asarray(goal["classes"])))[0]

    def output_scores(self, examples, split):
        """What each channel of the last layer writes to the true class of each example, against the mean class. Over a sample of the goal's examples."""
        labels = np.asarray(self._arr(split, "labels"))[np.asarray(examples)]
        hm = np.asarray(self._arr(split, "s4b1_hm"), dtype=np.float64)[np.asarray(examples)]
        W = (self.m.fc.weight - self.m.fc.weight.mean(0, keepdim=True)).detach().numpy().astype(np.float64)
        write = (W[labels] * (hm - hm.mean(0))).mean(0)
        return {f"s4b1:{c}": float(abs(write[c])) for c in range(hm.shape[1])}

    def battery(self):
        from networks.resnet18.battery import ITEMS
        return ITEMS

    def battery_values(self, node):
        """Largest value on each drawn image, as a share of the largest firing on the reading photos."""
        n = ev2.parse(node)
        P = self._p("read")
        raw = np.stack([np.asarray(im.convert("RGB")) for _, im in self.battery()])
        out = []
        for start in range(0, len(raw), BATCH):
            f = ev2.value_field(self.m, P.tensor(None, raw=raw[start:start + BATCH]), n).flatten(1).max(1).values
            out.append(f.numpy())
        v, _ = self.peaks(node, "pos", "read")
        return np.maximum(np.concatenate(out).astype(np.float64), 0) / max(float(v.max()), 1e-9)

    def python(self):
        return os.path.join(CURVES, ".venv", "bin", "python")

    def evidence_help(self):
        return "deep NODE (prints a PNG path, then open it with Read), inputs NODE, inputs NODE --band neg, readers NODE, bundles NODE, edit NODE, photo NODE RANK (saves its RANK-th strongest real photo as a PNG, 0 is the strongest, and says where the peak is. Edit the file with PIL, then probe it), probe NODE IMAGE..., acts PARENT CHILD, trace PHOTO_NUMBER, fv NODE --natural, classes WORD, near NODE (shows where this node fires, above strong places of its most similar nodes where it stays quiet. These are what your words must separate it from. Prints a PNG path, then open it with Read)"

    def n_positions(self, node):
        return ev2.grid(ev2.parse(node)["s"]) ** 2

    def has_inputs(self, node):
        return not node.startswith("s0")

    def input_kept(self, name):
        return name.endswith(" (skip)")

    def leak_patterns(self):
        return [r"\bs[0-4](b[01]i?)?\b", r"\b(unit|channel|node|neuron|filter|logit|layer|block|stage|kernel)\s*#?\s*\d+\b", r"\b\d{1,3}\s*:\s*\d{1,3}\b"]

    def words_of(self, name):
        if name.startswith("logit:"):
            return "the class " + rn.class_names()[int(name.split(":")[1])]
        return None

    def output_names(self):
        return [f"logit:{j}" for j in range(1000)]
