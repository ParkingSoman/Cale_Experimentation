"""Lineage track: Pythia adapter (plan ~/.claude/plans/okay-look-at-the-prancy-jellyfish.md, log LINEAGE-LOG.md).

Pythia (GPT-NeoX): parallel attention + MLP, LayerNorm, MLP = dense_4h_to_h(gelu(dense_h_to_4h(x))).
One input vector per neuron (row of W_in), no gate. HF config says hidden_act "gelu" (exact), mlx-lm uses
gelu_approx, so MLP.__call__ is patched to exact gelu here.
"""
import json, os, shutil, sys
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models import gpt_neox

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
HF = os.path.join(ROOT, "research", "clean", "hf", "pythia")
TREE_DATA = os.path.join(ROOT, "research", "tree", "data")
CODE = os.path.join(ROOT, "research", "clean", "data")
RES = os.path.join(ROOT, "results")
SEQ = 512
DEV_LAYERS = (2, 5, 8)      # dev layers for Pythia-160M
SEALED_LAYERS = (4, 7, 10)  # never touched until a keeper-style check

gpt_neox.MLP.__call__ = lambda self, x: self.dense_4h_to_h(nn.gelu(self.dense_h_to_4h(x)))

FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "model.safetensors"]


def fetch(size="160m", rev="main"):
    """Local folder with the checkpoint. 'main' is kept; other revisions go to a temp folder (delete after use)."""
    from huggingface_hub import hf_hub_download
    d = os.path.join(HF, f"{size}-{rev}")
    if not os.path.exists(os.path.join(d, "model.safetensors")):
        for f in FILES:
            hf_hub_download(f"EleutherAI/pythia-{size}", f, revision=rev, local_dir=d)
        shutil.rmtree(os.path.join(d, ".cache"), ignore_errors=True)
    return d


def drop(size, rev):
    if rev != "main":
        shutil.rmtree(os.path.join(HF, f"{size}-{rev}"), ignore_errors=True)


def load_pythia(size="160m", rev="main"):
    model, tok = load(fetch(size, rev))
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return model, tok


def layers(model):
    return model.model.h


def mlp_w(model, L):
    m = layers(model)[L].mlp
    return (np.array(m.dense_h_to_4h.weight), np.array(m.dense_h_to_4h.bias),
            np.array(m.dense_4h_to_h.weight), np.array(m.dense_4h_to_h.bias))


def blocks(tok, kind, n, skip=0):
    """n blocks of SEQ+1 ids. kind: 'train'/'test' (wikitext-2) or 'code_train'/'code_test'."""
    if kind.startswith("code"):
        text = open(os.path.join(CODE, kind + ".txt")).read()
    else:
        text = open(os.path.join(TREE_DATA, f"wikitext2_{kind}.txt")).read()
    need = (n + skip) * (SEQ + 1)
    ids, pos = [], 0
    while len(ids) < need and pos < len(text):
        ids += tok.encode(text[pos:pos + 200_000]); pos += 200_000
    ids = np.array(ids[skip * (SEQ + 1):need], dtype=np.int32)
    n = len(ids) // (SEQ + 1)
    return mx.array(ids[:n * (SEQ + 1)].reshape(n, SEQ + 1))


def block_ce(model, b, batch=4):
    out = []
    for i in range(0, b.shape[0], batch):
        x = b[i:i + batch]
        lg = model(x[:, :-1]); lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        out.append(np.array(-mx.take_along_axis(lg, x[:, 1:, None], axis=-1).squeeze(-1).mean(1)))
    return np.concatenate(out)


class Grab(nn.Module):
    def __init__(self, inner):
        super().__init__(); self.inner, self.x = inner, []

    def __call__(self, x):
        self.x.append(np.array(x).reshape(-1, x.shape[-1])); return self.inner(x)


def mlp_in(model, b, L, batch=4):
    """MLP input x (after post_attention_layernorm) at layer L, block position 0 dropped (sink)."""
    lay = layers(model)[L]; g = Grab(lay.mlp); lay.mlp = g
    for i in range(0, b.shape[0], batch):
        mx.eval(model(b[i:i + batch, :-1]))
    lay.mlp = g.inner
    X = np.concatenate(g.x, 0)
    return X[np.arange(X.shape[0]) % SEQ != 0]


def gelu(z):
    from scipy.special import erf
    return 0.5 * z * (1 + erf(z / np.sqrt(2)))


def boot(d, n=2000, seed=0):
    d = np.asarray(d); r = np.random.default_rng(seed)
    m = np.array([d[r.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return [float(d.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def save(name, obj):
    p = os.path.join(RES, f"lineage-{name}.json")
    json.dump(obj, open(p, "w"), indent=1)
    return p
