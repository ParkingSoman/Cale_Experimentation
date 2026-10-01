"""NestedGemma: one .nest file, two instant knobs (precision 4/8/16 bits, MLP width 100/90/85%).

    m = NestedGemma.load("models/g12-nest/model.nest", ram_budget_gb=12)
    m.tiers()                                  # bits x width: GB, resident or streamed, scores
    m.set(bits=8, width=0.90)                  # instant: no reload, no file reads
    out = m.generate("Explain X", bits=4)      # per-call override
    out = m.choose(prompt, ["yes", "no"], bits=16)

Every weight table becomes a PlaneLinear: y = row_scale * (x @ A) + x @ B + x @ C, with
planes dropped for lower bits (row_scale only applies from 8 bits up). The tier's norms
(QAT for 4 bits, original for 8 and 16) are swapped in by set(). Tensors that do not fit
the RAM budget are streamed from the file one layer ahead (nest/offload.py).
"""
import os
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from mlx_vlm.models.gemma4 import language as g4

from .nestfile import NestFile
from .offload import Prefetcher, TierStore, block_of, priority, table_of

PLANES = {4: ("A",), 8: ("A", "B"), 16: ("A", "B", "C")}
RESERVE_GB = 1.5          # activations, KV cache, MLX buffers
_ORIG_LAYER_CALL = g4.DecoderLayer.__call__


def _parent(root, path):
    *head, name = path.split(".")
    for h in head:
        root = getattr(root, h)
    return root, name


class PlaneLinear(nn.Module):
    """A weight table stored as precision planes, with an optional width (MLP neurons)."""

    def __init__(self, name, table, gq, bias=False):
        super().__init__()
        self.h = SimpleNamespace(name=name, kind=table["kind"], blocks=table["blocks"],
                                 out=table["shape"][0], gq=gq, res={}, cur={}, bits=4, width=None)
        if bias:
            self.bias = mx.zeros((table["shape"][0],), dtype=mx.bfloat16)

    def _arr(self, n):
        h = self.h
        a = h.res.get(n)
        return a if a is not None else h.cur[n]

    def _parts(self, plane, blk=None, rows=None):
        tag = plane if blk is None else f"{plane}{blk}"
        w, s, b = (self._arr(f"{self.h.name}|{tag}|{p}") for p in ("weight", "scales", "biases"))
        if rows is not None:
            w, s, b = w[:rows], s[:rows], b[:rows]
        return w, s, b

    def _qmm(self, x, plane, parts):
        g, bits = self.h.gq[plane]
        return mx.quantized_matmul(x, *parts, transpose=True, group_size=g, bits=bits)

    def __call__(self, x):
        h = self.h
        planes = PLANES[h.bits]
        rs = self._arr(f"{h.name}|B|row_scale") if h.bits >= 8 else None
        y = None
        if h.kind == "cols":
            nb = sum(1 for c in h.blocks[1:] if c <= (h.width or h.blocks[-1]))
            for plane in planes:
                yp = None
                for k in range(nb):
                    xk = x[..., h.blocks[k]:h.blocks[k + 1]]
                    t = self._qmm(xk, plane, self._parts(plane, k))
                    yp = t if yp is None else yp + t
                if plane == "A" and rs is not None:
                    yp = yp * rs
                y = yp if y is None else y + yp
        else:
            rows = h.width if h.kind == "rows" else None
            for plane in planes:
                yp = self._qmm(x, plane, self._parts(plane, rows=rows))
                if plane == "A" and rs is not None:
                    yp = yp * (rs if rows is None else rs[:rows])
                y = yp if y is None else y + yp
        if "bias" in self:
            y = y + self.bias
        return y


class PlaneEmbedding(PlaneLinear):
    """Word table: row lookup for inputs, and the tied output layer (as_linear)."""

    def __call__(self, ids):
        h = self.h
        y = None
        for plane in PLANES[h.bits]:
            w, s, b = self._parts(plane)
            g, bits = h.gq[plane]
            t = mx.dequantize(w[ids], s[ids], b[ids], group_size=g, bits=bits)
            if plane == "A" and h.bits >= 8:
                t = t * self._arr(f"{h.name}|B|row_scale")[ids][..., None]
            y = t if y is None else y + t
        return y

    def as_linear(self, x):
        return PlaneLinear.__call__(self, x)


def _layer_call(self, x, mask=None, cache=None, per_layer_input=None, shared_kv=None, offset=None):
    rt = getattr(self, "_nest", None)
    if rt is None:
        return _ORIG_LAYER_CALL(self, x, mask, cache, per_layer_input, shared_kv, offset)
    rt._enter(self._nest_idx)
    out = _ORIG_LAYER_CALL(self, x, mask, cache, per_layer_input, shared_kv, offset)
    mx.async_eval(out[0])
    rt._exit(self._nest_idx)
    return out


def _extract_files(nf, path):
    d = Path(path).parent / ".nest-files"
    d.mkdir(exist_ok=True)
    for name, t in nf.tensors.items():
        if t["region"] == "X":
            f = d / name.split("|", 1)[1]
            if not f.exists() or f.stat().st_size != t["nbytes"]:
                f.write_bytes(nf.blob(name))
    return d


def _skeleton(cfg_dir):
    from mlx_vlm.utils import (apply_generation_config_defaults, get_model_and_args, load_config,
                               load_image_processor, load_processor, update_module_configs)
    config = load_config(cfg_dir)
    model_class, _ = get_model_and_args(config=config)
    config.setdefault("text_config", config.pop("llm_config", {}))
    config.setdefault("vision_config", {})
    config.setdefault("audio_config", {})
    mc = model_class.ModelConfig.from_dict(config)
    mc = update_module_configs(mc, model_class, config, ["text", "vision", "perceiver", "projector", "audio"])
    mc = apply_generation_config_defaults(mc, config)
    model = model_class.Model(mc)
    proc = load_processor(cfg_dir, True, eos_token_ids=getattr(model.config, "eos_token_id", None))
    ip = load_image_processor(cfg_dir)
    if ip is not None:
        proc.image_processor = ip
    return model, proc


class NestedGemma:
    @classmethod
    def load(cls, path, ram_budget_gb=12.0, bits=4, width=1.0):
        self = cls()
        t0 = time.perf_counter()
        self.nf = nf = NestFile(path)
        H = nf.header
        self.model, self.processor = _skeleton(_extract_files(nf, path))
        self.store = TierStore(nf)
        self.prefetch = Prefetcher(self.store)
        gq = {p: tuple(v) for p, v in H["planes"].items()}

        # Replace every weight table with a plane module.
        self.mods = {}
        for m, t in H["tables"].items():
            parent, attr = _parent(self.model, m)
            old = getattr(parent, attr)
            if m.endswith("embed_tokens"):
                new = PlaneEmbedding(m, t, gq)
            else:
                new = PlaneLinear(m, t, gq, bias="bias" in old)
            setattr(parent, attr, new)
            self.mods[m] = new
        self.layers = self.model.language_model.model.layers
        for i, layer in enumerate(self.layers):
            layer._nest_idx = i
            layer._nest = None
        g4.DecoderLayer.__call__ = _layer_call

        # Norms and other small tensors for both tier families, always in RAM.
        self.norms = {"N4": {}, "N8": {}}
        for name, t in nf.tensors.items():
            if t["region"] in self.norms:
                self.norms[t["region"]][name.split("|")[0]] = self.store.read(name)

        # Residency: fill the RAM budget in priority order.
        budget = ram_budget_gb * 1e9 - RESERVE_GB * 1e9
        budget -= sum(t["nbytes"] for t in nf.tensors.values() if t["region"] in ("N4", "N8"))
        self.resident = set()
        used = 0
        for name in priority(H):
            t = nf.tensors[name]
            if t["region"] not in ("A", "B", "C"):
                continue
            if used + t["nbytes"] > budget:
                continue
            arr = self.store.read(name)
            self.mods[table_of(name)].h.res[name] = arr
            self.resident.add(name)
            used += t["nbytes"]
            if len(self.resident) % 64 == 0:
                mx.eval(list(self.mods[table_of(name)].h.res.values()))
        mx.eval([a for m in self.mods.values() for a in m.h.res.values()])
        emb = [n for n in H["tensors"] if n.startswith("language_model.model.embed_tokens|A|")]
        assert all(n in self.resident for n in emb), "RAM budget too small for the 4-bit word table"
        self.resident_bytes = used
        self.load_s = time.perf_counter() - t0
        self.bits = self.width = None
        self.set(bits=bits, width=width)
        return self

    # --- knobs -------------------------------------------------------------------------
    def _neurons(self, width):
        ws = self.nf.header["widths"]
        key = min(ws, key=lambda k: abs(int(k) / 100 - width))
        return int(key), ws[key]

    def needed(self, bits, width):
        """(name, rows) of every tensor the tier (bits, width) reads."""
        pct, n = self._neurons(width)
        out = []
        for m, mod in self.mods.items():
            h = mod.h
            names = []
            for plane in PLANES[bits]:
                if h.kind == "cols":
                    nb = sum(1 for c in h.blocks[1:] if c <= n)
                    names += [(f"{m}|{plane}{k}|{p}", None) for k in range(nb) for p in ("weight", "scales", "biases")]
                else:
                    rows = n if h.kind == "rows" else None
                    names += [(f"{m}|{plane}|{p}", rows) for p in ("weight", "scales", "biases")]
            if bits >= 8:
                names.append((f"{m}|B|row_scale", n if h.kind == "rows" else None))
            out += names
        return out

    def _bytes(self, name, rows):
        t = self.nf.tensors[name]
        return t["nbytes"] if rows is None else t["nbytes"] // t["shape"][0] * rows

    def set(self, bits=None, width=None):
        """Change precision and/or width. Returns milliseconds taken. Reads no file bytes."""
        t = time.perf_counter()
        bits = self.bits if bits is None else int(bits)
        width = self.width if width is None else float(width)
        assert bits in PLANES, "bits must be 4, 8 or 16"
        if bits == self.bits and width == self.width:
            return 0.0
        pct, n = self._neurons(width)
        for mod in self.mods.values():
            mod.h.bits = bits
            mod.h.width = n if mod.h.kind != "none" else None
        if self.bits is None or (bits >= 8) != (self.bits >= 8):
            self.model.update(tree_unflatten(list(self.norms["N8" if bits >= 8 else "N4"].items())))
        # Streaming plan: per layer, the needed tensors that are not resident.
        self.plan = {}
        for name, rows in self.needed(bits, width):
            if name in self.resident:
                continue
            m = table_of(name)
            layer = self.nf.header["tables"][m]["layer"]
            assert layer >= 0, f"non-layer tensor {name} must be resident; raise ram_budget_gb"
            self.plan.setdefault(layer, []).append((name, rows))
        self.prefetch.clear()
        for i, layer in enumerate(self.layers):
            layer._nest = self if i in self.plan else None
        self.streamed_layers = sorted(self.plan)
        self.bits, self.width = bits, width
        return (time.perf_counter() - t) * 1000

    def _enter(self, i):
        got = self.prefetch.get(i, self.plan[i])
        for m in self._layer_mods(i):
            m.h.cur = got
        sl = self.streamed_layers
        nxt = sl[(sl.index(i) + 1) % len(sl)]
        self.prefetch.start(nxt, self.plan[nxt])

    def _exit(self, i):
        for m in self._layer_mods(i):
            m.h.cur = {}

    def _layer_mods(self, i):
        cache = getattr(self, "_lm_cache", None)
        if cache is None:
            cache = self._lm_cache = {}
            for m, mod in self.mods.items():
                cache.setdefault(self.nf.header["tables"][m]["layer"], []).append(mod)
        return cache.get(i, [])

    # --- information ---------------------------------------------------------------------
    def tiers(self, scores=None):
        """One row per (bits, width): GB the tier reads, GB of it in RAM, GB streamed per pass."""
        rows = []
        for bits in (4, 8, 16):
            for key in sorted(self.nf.header["widths"], key=int, reverse=True):
                need = self.needed(bits, int(key) / 100)
                total = sum(self._bytes(n, r) for n, r in need)
                res = sum(self._bytes(n, r) for n, r in need if n in self.resident)
                tier = f"{bits}bit-w{key}"
                rows.append({"tier": tier, "bits": bits, "width": int(key) / 100,
                             "gb": round(total / 1e9, 2), "resident_gb": round(res / 1e9, 2),
                             "streamed_gb_per_pass": round((total - res) / 1e9, 2),
                             "scores": (scores or {}).get(tier)})
        return rows

    # --- calls ------------------------------------------------------------------------------
    def _call(self, fn, bits, width):
        prev = (self.bits, self.width)
        sw = self.set(bits, width)
        b0, r0, w0 = self.store.bytes, self.store.read_s, self.prefetch.wait_s
        t = time.perf_counter()
        try:
            out = fn()
        finally:
            dt = time.perf_counter() - t
        out["stats"] = {"tier": f"{self.bits}bit-w{self._neurons(self.width)[0]}", "switch_ms": round(sw, 3),
                        "seconds": round(dt, 3), "ssd_bytes": self.store.bytes - b0,
                        "ssd_read_s": round(self.store.read_s - r0, 3),
                        "ssd_wait_s": round(self.prefetch.wait_s - w0, 3), **out.pop("_extra", {})}
        if (bits is not None or width is not None) and (self.bits, self.width) != prev:
            self.set(*prev)
        return out

    @staticmethod
    def _rec(prompt):
        return prompt if isinstance(prompt, dict) else {"user": prompt}

    def generate(self, prompt, max_tokens=256, bits=None, width=None):
        from .gen import EOS, _prefill
        from .io import encode

        def run():
            ids, kw = encode(self.processor, self._rec(prompt), with_answer=False, max_tokens=4096)
            lm = self.model.language_model
            cache = lm.make_cache()
            t = time.perf_counter()
            hdn = _prefill(self.model, ids, kw, cache)
            tok = mx.argmax(lm.logits_from_hidden(hdn[:, -1:]), axis=-1)
            mx.eval(tok)
            t1 = time.perf_counter()
            out = []
            for _ in range(max_tokens):
                v = int(tok.item())
                if v in EOS:
                    break
                out.append(v)
                hdn = lm.model(tok.reshape(1, 1), cache=cache)
                tok = mx.argmax(lm.logits_from_hidden(hdn[:, -1:]), axis=-1)
                mx.async_eval(tok)
            t2 = time.perf_counter()
            return {"text": self.processor.tokenizer.decode(out, skip_special_tokens=True),
                    "_extra": {"prompt_tokens": int(ids.shape[1]), "tokens": len(out),
                               "prefill_s": round(t1 - t, 3),
                               "tok_per_s": round(len(out) / (t2 - t1), 2) if out and t2 > t1 else None}}

        return self._call(run, bits, width)

    def choose(self, prompt, options=None, bits=None, width=None):
        """One forward pass. prompt: a bench record with 'choices' letters, or a question
        plus a list of option strings (shown as A., B., ...)."""
        from .gen import ANSWER_PREFIX, _prefill
        from .io import encode
        rec = self._rec(prompt)
        if options is not None:
            letters = "ABCDEFGHIJ"[: len(options)]
            body = "\n".join(f"{c}. {o}" for c, o in zip(letters, options))
            rec = dict(rec, user=f"{rec['user']}\n\n{body}\n\nAnswer with the letter of the best option.",
                       choices=list(letters))
        tk = self.processor.tokenizer
        lid = {c: tk.encode(" " + c, add_special_tokens=False)[0] for c in rec["choices"]}

        def run():
            ids, kw = encode(self.processor, rec, with_answer=False, max_tokens=4096, answer_prefix=ANSWER_PREFIX)
            hdn = _prefill(self.model, ids, kw, None)
            lg = self.model.language_model.logits_from_hidden(hdn[:, -1])[0].astype(mx.float32)
            lp = lg - mx.logsumexp(lg)
            sc = {c: float(lp[lid[c]].item()) for c in rec["choices"]}
            best = max(sc, key=sc.get)
            res = {"choice": best, "scores": sc, "_extra": {"prompt_tokens": int(ids.shape[1])}}
            if options is not None:
                res["option"] = options[rec["choices"].index(best)]
            return res

        return self._call(run, bits, width)
