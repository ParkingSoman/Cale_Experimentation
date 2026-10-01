"""Run the bf16 12B one layer at a time over many samples.

The bf16 model (24 GB) does not fit in 24 GB of RAM. This runner loads one
decoder layer, pushes every sample's hidden state through it, frees the layer,
then loads the next. The file is read about once per full pass.

Use it for:
  - the true full-precision reference predictions
  - activation statistics on the unquantized weights

Do NOT call nest.elastic.patch() in the same process.
"""
import json
from collections import defaultdict

import mlx.core as mx
from mlx.utils import tree_unflatten
from mlx_vlm import load as vlm_load

from .io import encode

PREFIX = "language_model.model.layers."


class LayerwiseRunner:
    def __init__(self, src):
        self.src = src
        self.model, self.processor = vlm_load(src, lazy=True)
        self.lm = self.model.language_model.model
        wm = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
        self.layer_files = defaultdict(set)
        for k, f in wm.items():
            if k.startswith(PREFIX):
                self.layer_files[int(k[len(PREFIX):].split(".")[0])].add(f)

    def _fresh(self, i):
        """Lazy (not yet read) parameters of layer i. Replacing a layer's parameters
        with these drops the previous, already-read arrays and frees their memory."""
        p = f"{PREFIX}{i}."
        out = []
        for f in self.layer_files[i]:
            for k, v in mx.load(f"{self.src}/{f}").items():
                if k.startswith(p):
                    out.append((k[len(p):], v))
        return tree_unflatten(out)

    def prepare(self, recs):
        """Embed every sample and build its attention masks."""
        self.samples = []
        for r in recs:
            ids, kw = encode(self.processor, r)
            emb = self.model.get_input_embeddings(input_ids=ids, **kw).inputs_embeds
            masks = self.lm._make_masks(emb, [None] * len(self.lm.layers), kw.get("mm_token_type_ids"))
            mx.eval(emb)
            by_type = {}
            for layer, m in zip(self.lm.layers, masks):
                by_type[layer.layer_type] = m
            self.samples.append({"rec": r, "ids": ids, "kw": kw, "h": emb, "masks": by_type})
        return self.samples

    def run(self, layer_hook=None, skip=()):
        """Push all samples through all layers. layer_hook(i, sample, h_in, h_out) is optional."""
        for i, layer in enumerate(self.lm.layers):
            layer.update(self._fresh(i))
            for s in self.samples:
                h_in = s["h"]
                if i in skip:
                    h_out = h_in * layer.layer_scalar
                else:
                    h_out, _, _ = layer(h_in, s["masks"][layer.layer_type], None)
                mx.eval(h_out)
                if layer_hook is not None:
                    layer_hook(i, s, h_in, h_out)
                s["h"] = h_out
            layer.update(self._fresh(i))
            mx.clear_cache()

    def logprobs(self, s, positions):
        h = self.lm.norm(s["h"])[0][mx.array(positions)]
        lg = self.model.language_model.logits_from_hidden(h).astype(mx.float32)
        return lg - mx.logsumexp(lg, axis=-1, keepdims=True)
