"""NestedGemma: one .nest file, instant knobs: MLP width (100/90/85%) and the drafter (speed).

    m = NestedGemma.load("models/g12-nest/model.nest")
    m.tiers()                                     # width x drafter: GB, resident or streamed, scores
    m.set(width=0.90, draft=True)                 # instant: no reload, no file reads
    out = m.generate("Explain X", width=0.85)     # per-call override
    out = m.choose("Is X true?", ["yes", "no"])   # one forward pass

Every weight table becomes a NestLinear. MLP neurons are stored most important first,
so a width uses a row prefix of gate/up and the first column blocks of down_proj.
The drafter (Google's QAT MTP assistant) is stored in the same file. Tensors that do
not fit the RAM budget are streamed from the file one layer ahead (nest/offload.py).
"""
import io
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from mlx_vlm.models.gemma4 import language as g4

from .nestfile import NestFile
from .offload import Prefetcher, TierStore, priority, table_of

RESERVE_GB = 1.5          # activations, KV cache, MLX buffers
_ORIG_LAYER_CALL = g4.DecoderLayer.__call__


def _parent(root, path):
    *head, name = path.split(".")
    for h in head:
        root = root[int(h)] if h.isdigit() else getattr(root, h)
    return root, name


class NestLinear(nn.Module):
    """A 4-bit weight table with an optional width (number of MLP neurons used)."""

    def __init__(self, name, table, gq, bias=False):
        super().__init__()
        self.h = SimpleNamespace(name=name, kind=table["kind"], blocks=table["blocks"],
                                 gq=gq, res={}, cur={}, width=None, cache={}, cache_max=2)
        if bias:
            self.bias = mx.zeros((table["shape"][0],), dtype=mx.bfloat16)

    def _arr(self, n):
        a = self.h.res.get(n)
        return a if a is not None else self.h.cur[n]

    def _parts(self, tag, rows=None):
        w, s, b = (self._arr(f"{self.h.name}|{tag}|{p}") for p in ("weight", "scales", "biases"))
        return (w, s, b) if rows is None else (w[:rows], s[:rows], b[:rows])

    def _qmm(self, x, parts):
        g, bits = self.h.gq
        return mx.quantized_matmul(x, *parts, transpose=True, group_size=g, bits=bits)

    def colpart(self, n):
        """Joined down_proj table cut to its first n input columns. A strided cut is copied on
        every call, so the copy is made once per width and kept (the last two widths)."""
        h = self.h
        per, g = 32 // h.gq[1], h.gq[0]
        w, s, b = self._parts("A")
        if w.shape[1] * per == n:
            return w, s, b
        c = h.cache.pop(n, None)               # pop and re-insert: the dict stays in least-recently-used order
        if c is None:
            c = tuple(mx.contiguous(p[:, :k]) for p, k in ((w, n // per), (s, n // g), (b, n // g)))
            while len(h.cache) >= h.cache_max:
                h.cache.pop(next(iter(h.cache)))
        h.cache[n] = c
        return c

    def __call__(self, x):
        h = self.h
        if h.kind == "cols1":
            y = self._qmm(x, self.colpart(h.width or h.blocks[-1]))
        elif h.kind == "cols":
            nb = sum(1 for c in h.blocks[1:] if c <= (h.width or h.blocks[-1]))
            # Column blocks summed in bf16. Summing in float32 was tried: no closer to qg4's
            # single matmul in the model (MLX kernels sum in different orders anyway) and
            # 11% slower decode, 20% slower prefill at full width.
            y = None
            for k in range(nb):
                t = self._qmm(x[..., h.blocks[k]:h.blocks[k + 1]], self._parts(f"A{k}"))
                y = t if y is None else y + t
        else:
            y = self._qmm(x, self._parts("A", h.width if h.kind == "rows" else None))
        if "bias" in self:
            y = y + self.bias
        return y


class NestEmbedding(NestLinear):
    """Word table: row lookup for inputs, and the tied output layer (as_linear)."""

    def __call__(self, ids):
        w, s, b = self._parts("A")
        g, bits = self.h.gq
        return mx.dequantize(w[ids], s[ids], b[ids], group_size=g, bits=bits)

    def as_linear(self, x):
        return NestLinear.__call__(self, x)


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
    """Small files (config, tokenizer, chat template) go to a folder next to the nest file.
    The drafter weights stay in the nest file and are loaded from memory."""
    d = Path(path).parent / ".nest-files"
    for name, t in nf.tensors.items():
        if t["region"] != "X" or name.endswith(".safetensors"):
            continue
        f = d / name.split("|", 1)[1]
        f.parent.mkdir(parents=True, exist_ok=True)
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


def _load_drafter(nf, d):
    """Load the drafter with mlx-vlm, feeding its weights from the nest file instead of disk."""
    import mlx_vlm.utils as U
    from mlx_vlm.speculative.drafters import load_drafter
    blob = nf.blob("file|drafter/model.safetensors")
    stub = d / "drafter" / "model.safetensors"
    stub.write_bytes(b"")                  # mlx-vlm looks for a .safetensors file; contents come from the blob
    orig = U._load_safetensors
    U._load_safetensors = lambda path: mx.load(io.BytesIO(blob), format="safetensors")
    try:
        return load_drafter(str(d / "drafter"))
    finally:
        U._load_safetensors = orig


class NestedGemma:
    @classmethod
    def load(cls, path, ram_budget_gb=None, width=1.0, draft=False, drafter=True):
        """ram_budget_gb=None keeps everything in RAM. A smaller budget keeps the most
        important blocks in RAM (narrow-width heads first) and streams the rest."""
        self = cls()
        t0 = time.perf_counter()
        self.nf = nf = NestFile(path)
        H = nf.header
        d = _extract_files(nf, path)
        self.model, self.processor = _skeleton(d)
        self.store = TierStore(nf)
        self.prefetch = Prefetcher(self.store)
        gq = tuple(H["planes"]["A"])

        self.mods = {}
        for m, t in H["tables"].items():
            parent, attr = _parent(self.model, m)
            old = getattr(parent, attr)
            new = NestEmbedding(m, t, gq) if m.endswith("embed_tokens") else NestLinear(m, t, gq, bias="bias" in old)
            setattr(parent, attr, new)
            self.mods[m] = new
        self.layers = self.model.language_model.model.layers
        for i, layer in enumerate(self.layers):
            layer._nest_idx = i
            layer._nest = None
        g4.DecoderLayer.__call__ = _layer_call

        norms = {n.split("|")[0]: self.store.read(n) for n, t in nf.tensors.items() if t["region"] == "N4"}
        self.model.update(tree_unflatten(list(norms.items())))

        budget = float("inf") if ram_budget_gb is None else ram_budget_gb * 1e9 - RESERVE_GB * 1e9
        self.resident, used = set(), 0
        for name in priority(H):
            t = nf.tensors[name]
            if t["region"] != "A" or used + t["nbytes"] > budget:
                continue
            self.mods[table_of(name)].h.res[name] = self.store.read(name)
            self.resident.add(name)
            used += t["nbytes"]
        emb = [n for n in H["tensors"] if n.startswith("language_model.model.embed_tokens|")]
        assert all(n in self.resident for n in emb), "RAM budget too small for the word table"
        self.resident_bytes = used
        # Fully resident: any width from MIN_WIDTH to 1.0, in steps of 64 neurons.
        self.any = ram_budget_gb is None
        self.inter = H["intermediate_size"]
        self.cap_n = self.inter
        if self.any:
            self._join_down_blocks()
        self.drafter = self.draft_kind = None
        if drafter:
            self.drafter, self.draft_kind = _load_drafter(nf, d)
        self.load_s = time.perf_counter() - t0
        self.width = None
        self.draft = False
        self._cap_dirty = False
        self.set(width=width, draft=draft)
        return self

    def _join_down_blocks(self):
        """The file stores down_proj as three column blocks (exact cuts of one table). Join them
        back into one table so any width is a plain column prefix. Block edges are multiples of
        the group size, so the join is exact. One table at a time keeps the extra memory small."""
        for m, mod in self.mods.items():
            h = mod.h
            if h.kind != "cols":
                continue
            nb = len(h.blocks) - 1
            for p in ("weight", "scales", "biases"):
                parts = [h.res.pop(f"{m}|A{k}|{p}") for k in range(nb)]
                h.res[f"{m}|A|{p}"] = mx.concatenate(parts, axis=1)
                self.resident.difference_update(f"{m}|A{k}|{p}" for k in range(nb))
                self.resident.add(f"{m}|A|{p}")
            mx.eval(*[h.res[f"{m}|A|{p}"] for p in ("weight", "scales", "biases")])
            h.kind = "cols1"
        mx.clear_cache()

    # --- knobs -------------------------------------------------------------------------
    MIN_WIDTH = 0.5
    STEP = 64

    def _neurons(self, width):
        """(percent, neuron count) for a width. Fully resident models take any width."""
        if self.any:
            n = int(round(min(max(float(width), self.MIN_WIDTH), 1.0) * self.inter / self.STEP)) * self.STEP
            n = min(n, self.cap_n)
            return round(100 * n / self.inter), n
        ws = self.nf.header["widths"]
        key = min(ws, key=lambda k: abs(int(k) / 100 - width))
        return int(key), ws[key]

    def needed(self, width):
        """(name, rows) of every tensor the width reads."""
        _, n = self._neurons(width)
        out = []
        for m, mod in self.mods.items():
            h = mod.h
            if h.kind == "cols":
                nb = sum(1 for c in h.blocks[1:] if c <= n)
                out += [(f"{m}|A{k}|{p}", None) for k in range(nb) for p in ("weight", "scales", "biases")]
            else:
                rows = n if h.kind == "rows" else None
                out += [(f"{m}|A|{p}", rows) for p in ("weight", "scales", "biases")]
        return out

    def _bytes(self, name, rows):
        t = self.nf.tensors[name]
        return t["nbytes"] if rows is None else t["nbytes"] // t["shape"][0] * rows

    def set(self, width=None, draft=None):
        """Change width and/or drafter use. Returns milliseconds taken. Reads no file bytes."""
        t = time.perf_counter()
        if draft is not None:
            assert not draft or self.drafter is not None, "loaded without the drafter"
            self.draft = bool(draft)
        if width is None:
            width = self.width
        elif isinstance(width, (list, tuple)):
            assert self.any, "per-layer widths need a fully resident model"
            assert len(width) == len(self.layers), f"need {len(self.layers)} widths"
            width = tuple(float(w) for w in width)
        else:
            width = float(width)
        if width != self.width or self._cap_dirty:
            self._cap_dirty = False
            per_layer = isinstance(width, tuple)
            for m, mod in self.mods.items():
                if mod.h.kind == "none":
                    mod.h.width = None
                else:
                    w = width[self.nf.header["tables"][m]["layer"]] if per_layer else width
                    mod.h.width = self._neurons(w)[1]
            self.plan = {}
            if self.any:
                self._build_caches()
            else:
                for name, rows in self.needed(width):
                    if name not in self.resident:
                        layer = self.nf.header["tables"][table_of(name)]["layer"]
                        self.plan.setdefault(layer, []).append((name, rows))
            self.prefetch.clear()
            for i, layer in enumerate(self.layers):
                layer._nest = self if i in self.plan else None
            self.streamed_layers = sorted(self.plan)
            self.width = width
        return (time.perf_counter() - t) * 1000

    def _build_caches(self):
        """Make the cut copies of down_proj for the new width now, so the first token is not slow."""
        arrs = []
        for mod in self.mods.values():
            if mod.h.kind == "cols1":
                arrs += list(mod.colpart(mod.h.width or mod.h.blocks[-1]))
        if arrs:
            mx.eval(*arrs)

    def set_cap(self, width):
        """Limit how much of every MLP table stays in RAM (fully resident models only).
        Shrinking copies a prefix of one layer at a time, then drops the old table, so the extra
        memory at any moment is one table. Growing reads the full tables back from the file.
        Requests wider than the cap run at the cap. Returns the new cap in neurons.
        If it fails half way (for example out of memory) the tables are mixed: the caller must
        drop the model and load it again (Engine does)."""
        assert self.any, "set_cap needs a fully resident model"
        n = int(round(min(max(float(width), self.MIN_WIDTH), 1.0) * self.inter / self.STEP)) * self.STEP
        if n == self.cap_n:
            return n
        for m, mod in self.mods.items():
            h = mod.h
            if h.kind not in ("rows", "cols1"):
                continue
            names = [f"{m}|A|{p}" for p in ("weight", "scales", "biases")]
            if n < self.cap_n:
                per, g = 32 // h.gq[1], h.gq[0]
                new = {}
                for nm, p in zip(names, ("weight", "scales", "biases")):
                    a = h.res[nm]
                    k = n if h.kind == "rows" else n // (per if p == "weight" else g)
                    new[nm] = mx.contiguous(a[:k] if h.kind == "rows" else a[:, :k])
            else:
                new = self._read_table(m, h, n)
            mx.eval(*new.values())
            h.res.update(new)
            h.cache.clear()
            h.cache_max = 2 if n >= self.inter else 1      # under a cap keep memory low: one cut copy only
        mx.clear_cache()
        self.cap_n = n
        self._cap_dirty = True
        self.set(self.width)
        return n

    def _read_table(self, m, h, n):
        """Read one table from the file, cut to its first n neurons."""
        per, g = 32 // h.gq[1], h.gq[0]
        out = {}
        for p in ("weight", "scales", "biases"):
            if h.kind == "rows":
                a = self.store.read(f"{m}|A|{p}", n)
            else:
                nb = len(h.blocks) - 1
                a = mx.concatenate([self.store.read(f"{m}|A{k}|{p}") for k in range(nb)], axis=1)
                a = mx.contiguous(a[:, : n // (per if p == "weight" else g)])
            out[f"{m}|A|{p}"] = a
        return out

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

    def tier_name(self, width=None, draft=None):
        width = self.width if width is None else width
        w = "custom" if isinstance(width, tuple) else self._neurons(width)[0]
        return f"w{w}" + ("+draft" if (self.draft if draft is None else draft) else "")

    # --- information ---------------------------------------------------------------------
    def width_gb(self, width):
        """GB of weights one pass reads at this width (fully resident models)."""
        n = self._neurons(width)[1]
        tot = 0
        for name, t in self.nf.tensors.items():
            if t["region"] != "A":
                continue
            kind = self.mods[table_of(name)].h.kind
            tot += t["nbytes"] * n / self.inter if kind in ("rows", "cols1") else t["nbytes"]
        return tot / 1e9

    def tiers(self, scores=None):
        """One row per width: GB the width reads per pass, GB of it in RAM, GB streamed."""
        rows = []
        if self.any:
            for pct in (100, 90, 80, 70, 60, 50):
                gb = round(self.width_gb(pct / 100), 2)
                rows.append({"tier": f"w{pct}", "width": pct / 100, "gb": gb, "resident_gb": gb,
                             "streamed_gb_per_pass": 0.0, "drafter": self.drafter is not None,
                             "scores": (scores or {}).get(f"w{pct}")})
            return rows
        for key in sorted(self.nf.header["widths"], key=int, reverse=True):
            need = self.needed(int(key) / 100)
            total = sum(self._bytes(n, r) for n, r in need)
            res = sum(self._bytes(n, r) for n, r in need if n in self.resident)
            rows.append({"tier": f"w{key}", "width": int(key) / 100, "gb": round(total / 1e9, 2),
                         "resident_gb": round(res / 1e9, 2), "streamed_gb_per_pass": round((total - res) / 1e9, 2),
                         "drafter": self.drafter is not None, "scores": (scores or {}).get(f"w{key}")})
        return rows

    # --- calls ------------------------------------------------------------------------------
    def _call(self, fn, width, draft):
        prev = (self.width, self.draft)
        sw = self.set(width, draft)
        b0, w0 = self.store.bytes, self.prefetch.wait_s
        t = time.perf_counter()
        try:
            out = fn()
        finally:
            dt = time.perf_counter() - t
            if (self.width, self.draft) != prev:
                tier = self.tier_name()
                self.set(*prev)
            else:
                tier = self.tier_name()
        out["stats"] = {"tier": tier, "switch_ms": round(sw, 3), "seconds": round(dt, 3),
                        "ssd_bytes": self.store.bytes - b0, "ssd_wait_s": round(self.prefetch.wait_s - w0, 3),
                        **out.pop("_extra", {})}
        return out

    @staticmethod
    def _rec(prompt):
        return prompt if isinstance(prompt, dict) else {"user": prompt}

    def generate(self, prompt, max_tokens=256, width=None, draft=None):
        """Greedy generation. With the drafter, output can differ slightly from plain greedy
        (summation order in the verify pass); images run without the drafter."""
        from .gen import EOS, _prefill
        from .io import encode, messages
        rec = self._rec(prompt)

        def plain():
            ids, kw = encode(self.processor, rec, with_answer=False, max_tokens=4096)
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

        def drafted():
            from mlx_vlm import generate as vlm_generate
            text = self.processor.apply_chat_template(messages(rec, with_answer=False), tokenize=False,
                                                      add_generation_prompt=True)
            r = vlm_generate(self.model, self.processor, text, max_tokens=max_tokens, temperature=0.0,
                             verbose=False, draft_model=self.drafter, draft_kind=self.draft_kind)
            return {"text": r.text, "_extra": {"prompt_tokens": int(r.prompt_tokens), "tokens": int(r.generation_tokens),
                                               "tok_per_s": round(r.generation_tps, 2)}}

        use_draft = self.draft if draft is None else draft
        return self._call(drafted if use_draft and not rec.get("image") else plain, width,
                          False if rec.get("image") else draft)

    def choose(self, prompt, options=None, width=None):
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

        return self._call(run, width, False)

    # --- chat (used by the Cale server) ---------------------------------------------------
    def chat_stream(self, messages, max_tokens=256, width=None, draft=None, temperature=0.0):
        """Multi-turn text chat. Yields {"delta": str} pieces, then {"done": stats}.
        messages: [{"role": "system"|"user"|"assistant", "content": str}].
        With the drafter (greedy only) tokens arrive in small bursts."""
        from .gen import EOS, _prefill
        sw = self.set(width, draft)
        tok = self.processor.tokenizer
        msgs = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        use_draft = bool(self.draft and self.drafter is not None and temperature <= 0)
        t0 = time.perf_counter()
        n_prompt = n_out = 0
        if use_draft:
            from mlx_vlm.generate import stream_generate
            for r in stream_generate(self.model, self.processor, text, max_tokens=max_tokens, temperature=0.0,
                                     draft_model=self.drafter, draft_kind=self.draft_kind):
                if getattr(r, "is_draft", False):
                    continue
                if r.text:
                    yield {"delta": r.text}
                n_prompt, n_out = int(r.prompt_tokens), int(r.generation_tokens)
        else:
            enc = self.processor(text=text, images=None, return_tensors="np")
            ids = mx.array(enc["input_ids"])
            kw = {k: mx.array(enc[k]) for k in ("mm_token_type_ids",) if k in enc and enc[k] is not None}
            n_prompt = int(ids.shape[1])
            lm = self.model.language_model
            cache = lm.make_cache()

            def pick(h):
                lg = lm.logits_from_hidden(h[:, -1:])
                if temperature <= 0:
                    return mx.argmax(lg, axis=-1)
                return mx.random.categorical(lg / temperature, axis=-1)

            nxt = pick(_prefill(self.model, ids, kw, cache))
            mx.eval(nxt)
            out, sent = [], ""
            for _ in range(max_tokens):
                v = int(nxt.item())
                if v in EOS:
                    break
                out.append(v)
                hdn = lm.model(nxt.reshape(1, 1), cache=cache)
                nxt = pick(hdn)
                mx.async_eval(nxt)
                s = tok.decode(out, skip_special_tokens=True)
                if not s.endswith("�") and len(s) > len(sent):
                    yield {"delta": s[len(sent):]}
                    sent = s
            s = tok.decode(out, skip_special_tokens=True)
            if len(s) > len(sent):
                yield {"delta": s[len(sent):]}
            n_out = len(out)
        dt = time.perf_counter() - t0
        yield {"done": {"tier": self.tier_name(), "size_pct": None if isinstance(self.width, tuple) else self._neurons(self.width)[0],
                        "switch_ms": round(sw, 1), "seconds": round(dt, 3), "prompt_tokens": n_prompt,
                        "tokens": n_out, "finish": "length" if n_out >= max_tokens else "stop",
                        "drafter": use_draft}}
