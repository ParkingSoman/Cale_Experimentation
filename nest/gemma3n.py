"""Nested Gemma 3n: one loaded E4B that also runs as E2B and every official size in between.

Google trained these sizes inside E4B (MatFormer). A size is two choices per layer:
the FFN is half width (the first 8192 channels) or full width (16384), and layers 20-24 are
skipped or not. `nest/gemma3n_levels.json` holds Google's 16 official rows.

Switching a size only changes which blocks run. Nothing is copied or reloaded (about 1 ms).
Only text is supported. The vision and audio towers are dropped at load to save RAM.
"""
import gc
import json
import secrets
import struct
import time
from pathlib import Path

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.gemma3n.language import gelu_topk, logit_softcap

from .elastic import _full, _mm, _rows
from .levels3n import BLOCK, MAX_CONTEXT, load_levels, pick
from .cancel import Cancelled
from .stops import StopFilter

EOS = {1, 106}                  # <eos>, <end_of_turn>
STEP_TOKENS = 8                 # how often a size controller may change the size, in answer tokens
PREFILL_STEP = 2048             # the prompt is read in pieces of at most this many tokens, so memory grows with the prompt, not with its square.
                                # Never more than the model's local attention window (512): a piece longer than the window made the model lose facts in a long prompt
GROUP, BITS, MODE = 64, 4, "affine"     # how the FFN matrices of models/gemma-3n-e4b-4bit are quantized



def memory_bandwidth():
    """Bytes per second this computer reads from memory (the GPU's, for MLX). About 0.1 s, no model needed. Answer speed is bound by it."""
    a = mx.ones((256 * 1024 * 1024,), dtype=mx.float16)           # 512 MB, bigger than any cache
    mx.eval(a)
    best = 0.0
    for _ in range(4):
        t = time.perf_counter()
        mx.eval(mx.sum(a))
        best = max(best, a.nbytes / (time.perf_counter() - t))
    del a
    mx.clear_cache()
    return best


class _Skipped(nn.Module):
    """Stands in for a skipped layer. Skipped layers own no KV cache, so passing h through is exact."""

    def __call__(self, x, mask=None, cache=None, per_layer_input=None):
        return x


class _Parts:
    """Plain holder for one layer's FFN weights. Not a Module, so it never shows up as model parameters."""

    def __init__(self, full, a):
        self.full, self.a = full, a


def _own(parts):
    """Copy every array in matmul inputs, so the copy does not keep a bigger buffer alive."""
    return tuple(mx.contiguous(p) if isinstance(p, mx.array) else p for p in parts)


class ElasticMLP3n(nn.Module):
    """GeGLU MLP that runs on the first `blocks` blocks (1 or 2) of 8192 channels.

    The FFN of one layer is in one of three states (`ffn_state`):
      2  full: the original gate/up/down matrices (bit-identical to plain Gemma 3n E4B) plus a contiguous
         copy of the first 8192 columns of down. The half width reads the first 8192 rows of gate/up as free views.
      1  half: only the half-width pieces, as owned copies (about half the bytes of state 2).
      0  none: nothing. Used by layers that every held size skips.
    A call may run a layer in a state below the one it needs only after `restore` read the weights back.
    """

    def __init__(self, mlp):
        super().__init__()
        self.activation_sparsity = mlp.activation_sparsity
        if self.activation_sparsity > 0:
            self._std_multiplier = mlp._std_multiplier
        gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
        full = (_full(gate), _full(up), _full(down))
        half = (_rows(gate, BLOCK), _rows(up, BLOCK), self._first_cols(down, BLOCK))
        mx.eval([p for parts in half for p in parts if isinstance(p, mx.array)])
        self._p = _Parts(full, half)
        self.ffn_state = 2
        self.blocks = 2

    def set_blocks(self, blocks):
        if blocks > self.ffn_state:
            raise RuntimeError(f"FFN is in state {self.ffn_state}, size needs {blocks}")
        self.blocks = blocks

    @staticmethod
    def _first_cols(lin, n):
        """First n input columns of a (possibly quantized) Linear, as contiguous matmul inputs."""
        if isinstance(lin, nn.QuantizedLinear):
            per, g = 32 // lin.bits, lin.group_size
            return (mx.contiguous(lin.weight[:, : n // per]), mx.contiguous(lin.scales[:, : n // g]),
                    mx.contiguous(lin.biases[:, : n // g]), g, lin.bits, lin.mode)
        return mx.contiguous(lin.weight[:, :n]), None, None, None, None, None

    # --- residency ---------------------------------------------------------------------------
    def drop_to(self, state):
        """Free weights down to `state` (0 or 1). A state-1 layer owns its copies, so the full matrices are freed."""
        if state >= self.ffn_state:
            return
        if state == 1:
            half = tuple(_own(parts) for parts in self._p.a)
            mx.eval([p for parts in half for p in parts if isinstance(p, mx.array)])
            self._p = _Parts(None, half)
        else:
            self._p = _Parts(None, None)
        self.ffn_state = state
        self.blocks = min(self.blocks, state)

    def restore(self, state, tensors, prefix):
        """Read the weights back from `tensors` (a lazy dict from mx.load) to reach `state` (1 or 2).
        Returns the bytes read."""
        if state <= self.ffn_state:
            return 0
        read = 0

        def lin(name, quant_cls=None):
            nonlocal read
            w, s, b = (tensors[f"{prefix}.{name}.{k}"] for k in ("weight", "scales", "biases"))
            mx.eval(w, s, b)
            read += w.nbytes + s.nbytes + b.nbytes
            return (w, s, b, GROUP, BITS, MODE)

        if state == 2:
            gate, up, down = lin("gate_proj"), lin("up_proj"), lin("down_proj")
            half_down = self._p.a[2] if self.ffn_state == 1 else self._first_cols_parts(down)
            half = (tuple(p[:BLOCK] if isinstance(p, mx.array) else p for p in gate[:3]) + gate[3:],
                    tuple(p[:BLOCK] if isinstance(p, mx.array) else p for p in up[:3]) + up[3:], half_down)
            mx.eval([p for parts in half for p in parts if isinstance(p, mx.array)])
            self._p = _Parts((gate, up, down), half)
        else:
            gate, up = lin("gate_proj"), lin("up_proj")
            half_gate = _own(tuple(p[:BLOCK] if isinstance(p, mx.array) else p for p in gate[:3]) + gate[3:])
            half_up = _own(tuple(p[:BLOCK] if isinstance(p, mx.array) else p for p in up[:3]) + up[3:])
            half_down = self._first_cols_parts(lin("down_proj"))
            mx.eval([p for parts in (half_gate, half_up, half_down) for p in parts if isinstance(p, mx.array)])
            self._p = _Parts(None, (half_gate, half_up, half_down))
        self.ffn_state = state
        return read

    @staticmethod
    def _first_cols_parts(parts, n=BLOCK):
        w, s, b, g, bits, mode = parts
        per = 32 // bits
        out = (mx.contiguous(w[:, : n // per]), mx.contiguous(s[:, : n // g]), mx.contiguous(b[:, : n // g]), g, bits, mode)
        mx.eval(out[:3])
        return out

    def state_bytes(self):
        """Bytes this layer's FFN holds in state 0, 1 and 2. Call while the layer is in state 2."""
        half = sum(p.nbytes for parts in self._p.a for p in parts if isinstance(p, mx.array))
        return [0, half, self.nbytes()]

    def read_bytes(self):
        """Bytes read from disk to bring this layer's FFN from state 0 or 1 to state 2 (the three full matrices)."""
        return sum(p.nbytes for parts in self._p.full for p in parts if isinstance(p, mx.array))

    def nbytes(self):
        """Real bytes held. With the full matrices in RAM, the half-width gate/up are views of them and cost nothing."""
        if self._p.a is None:
            return 0
        groups = (*self._p.full, self._p.a[2]) if self._p.full is not None else self._p.a
        return sum(p.nbytes for parts in groups for p in parts if isinstance(p, mx.array))

    def __call__(self, x):
        parts = self._p.full if self.blocks == 2 else self._p.a
        gate_p, up_p, down_p = parts
        gate = _mm(x, gate_p)
        act = gelu_topk(gate, self._std_multiplier) if self.activation_sparsity > 0 else nn.gelu_approx(gate)
        return _mm(act * _mm(x, up_p), down_p)


class MappedEmbedding(nn.Module):
    """The per-layer embedding table (1.33 GB at 4 bits), read from the model file instead of held in RAM.

    A token needs only its own row (about 5 KB). The file is memory-mapped, so the Mac keeps recently used rows
    in its file cache and drops the rest whenever it needs the memory. The numbers are the same as the in-RAM
    table: the rows are dequantized with the same call.
    """

    def __init__(self, path, name, group_size, bits):
        super().__init__()
        self.group_size, self.bits = group_size, bits
        self._maps = {}
        for f in sorted(Path(path).glob("*.safetensors")):
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
            for part in ("weight", "scales", "biases"):
                t = header.get(f"{name}.{part}")
                if t:
                    dtype = np.uint32 if t["dtype"] == "U32" else np.uint16      # bfloat16 travels as raw 16-bit
                    self._maps[part] = np.memmap(f, dtype=dtype, mode="r", offset=8 + n + t["data_offsets"][0],
                                                 shape=tuple(t["shape"]))
        assert len(self._maps) == 3, f"{name} not found in {path}"

    def __call__(self, x):
        shape = x.shape
        ids = np.asarray(x).reshape(-1)
        w = mx.array(self._maps["weight"][ids])
        sc = mx.array(self._maps["scales"][ids]).view(mx.bfloat16)
        bi = mx.array(self._maps["biases"][ids]).view(mx.bfloat16)
        out = mx.dequantize(w, sc, bi, group_size=self.group_size, bits=self.bits)
        return out.reshape(*shape, -1)


class NestedGemma3n:
    @classmethod
    def load(cls, path, width=1.0, draft=False, text_only=True, loaded=None, ple_on_disk=False):
        from mlx_vlm import load as vlm_load
        self = cls()
        self.path = str(path)
        self.ple_on_disk = ple_on_disk
        self.model, self.processor = loaded or vlm_load(str(path))
        lm = self.model.language_model
        self.lm, self.layers = lm, lm.model.layers
        cfg = lm.config
        self.n_layers, self.first_shared = cfg.num_hidden_layers, lm.model.first_kv_shared_layer_idx
        for layer in self.layers:
            layer.mlp = ElasticMLP3n(layer.mlp)
        if text_only:
            self.model.vision_tower = None
            self.model.audio_tower = None
        if ple_on_disk:
            q = lm.model.embed_tokens_per_layer
            lm.model.embed_tokens_per_layer = MappedEmbedding(
                path, "language_model.model.embed_tokens_per_layer", q.group_size, q.bits)
            del q
        mx.eval(self.model.parameters())      # MLX arrays that are still lazy belong to the thread that loads them
        gc.collect()
        mx.clear_cache()
        self.orig_layers = list(self.layers)
        self.resident = [2] * self.n_layers
        self.layer_bytes = [l.mlp.state_bytes() for l in self.orig_layers]       # bytes in state 0, 1, 2
        self.layer_read = [l.mlp.read_bytes() for l in self.orig_layers]          # bytes a disk read costs per layer
        self.levels = [l for l in load_levels(self.n_layers) if min(l["skip"], default=99) >= self.first_shared]
        self.cap = 1.0
        self.draft = False                 # no drafter for Gemma 3n yet; kept so the Cale engine interface matches
        self.level = None
        self.set(width)
        self.generate("Hi", 2)                 # evaluates any constant that is still lazy, in this thread: MLX arrays belong to the thread that evaluates them
        return self

    # --- residency: which FFN weights sit in RAM -------------------------------------------
    def need(self, lv):
        """FFN state each layer needs to run size `lv`: 0 skipped, 1 half width, 2 full width."""
        return [0 if i in lv["skip"] else lv["blocks"][i] for i in range(self.n_layers)]

    def reside(self, levels):
        """Keep in RAM exactly what the union of `levels` needs. Frees the rest. Returns bytes freed."""
        return self.reside_states([max(col) for col in zip(*[self.need(self.resolve(l)) for l in levels])],
                                  first=self.resolve(levels[0]))

    def reside_states(self, states, first=None):
        """Set the FFN state of every layer (0 none, 1 half, 2 full): free what is above, read back what is below."""
        before = self.ffn_bytes()
        self.resident = list(states)
        for layer, st in zip(self.orig_layers, self.resident):
            layer.mlp.drop_to(st)
        self._read_back([(i, st) for i, st in enumerate(self.resident) if st > self.orig_layers[i].mlp.ffn_state])
        if first is not None:
            self.set(first)
        gc.collect()
        mx.clear_cache()
        return before - self.ffn_bytes()

    def missing(self, lv):
        """Layers (index, state needed) whose FFN is below what size `lv` needs."""
        return [(i, n) for i, n in enumerate(self.need(lv)) if n > self.orig_layers[i].mlp.ffn_state]

    def ensure(self, lv):
        """Read missing FFN weights from disk so size `lv` can run. Returns (bytes read, seconds)."""
        return self._read_back(self.missing(lv))

    def _read_back(self, todo):
        if not todo:
            return 0, 0.0
        t, read = time.perf_counter(), 0
        tensors = {}
        for f in sorted(Path(self.path).glob("*.safetensors")):
            tensors.update(mx.load(str(f)))
        for i, n in todo:
            read += self.orig_layers[i].mlp.restore(n, tensors, f"language_model.model.layers.{i}.mlp")
        del tensors
        return read, time.perf_counter() - t

    def settle(self):
        """Go back to the resident state after a call that read extra weights."""
        changed = False
        for layer, st in zip(self.orig_layers, self.resident):
            if layer.mlp.ffn_state > st:
                layer.mlp.drop_to(st)
                changed = True
        if changed:
            gc.collect()
            mx.clear_cache()

    def ffn_bytes(self):
        return sum(l.mlp.nbytes() for l in self.orig_layers)

    # --- what the Cale engine asks every backend (the llama.cpp model offers the same, see cale/backends) ---
    def active_bytes(self):
        return mx.get_active_memory()

    def clear_cache(self):
        self._pc = None
        gc.collect()
        mx.clear_cache()

    def all_full(self):
        return all(l.mlp.ffn_state == 2 for l in self.orig_layers)

    def model_files(self):
        return sorted(Path(self.path).glob("*.safetensors"))

    def prompt_len(self, messages):
        return int(self.prompt_ids(messages).shape[1])

    def close(self):
        self.model = self.lm = self.layers = self.orig_layers = None
        self.clear_cache()

    def size_bytes(self, lv):
        """FFN bytes held when exactly size `lv` is resident (measured on real arrays, from this model's layers)."""
        return sum(self.layer_bytes[i][n] for i, n in enumerate(self.need(lv)))

    # --- sizes ---------------------------------------------------------------------------
    def resolve(self, width):
        """A size from a level id ('e2b'), a level dict, or a fraction of E4B (0.5 = nearest to 50%)."""
        if isinstance(width, dict):
            return pick(self.levels, width["pct"], 100 * self.cap) if width["pct"] > 100 * self.cap + 0.5 else width
        if isinstance(width, str):
            return next(l for l in self.levels if l["id"] == width)
        return pick(self.levels, 100 * float(width), 100 * self.cap)

    def set(self, width=None, draft=None):
        """Switch size. Returns the switch time in ms."""
        t = time.perf_counter()
        if width is not None:
            lv = self.resolve(width)
            for i, layer in enumerate(self.orig_layers):
                self.layers[i] = _Skipped() if i in lv["skip"] else layer
                layer.mlp.set_blocks(0 if i in lv["skip"] else lv["blocks"][i])
            self.level = lv
        return (time.perf_counter() - t) * 1000

    def set_cap(self, width):
        """Largest size allowed (memory pressure). Lowers compute only: all weights stay in RAM.
        Only stores the cap. Every call resolves its size under the cap and reads any missing weights first,
        so nothing here may change the running size (a held size may not hold the weights that size needs)."""
        self.cap = float(width)

    def tier_name(self, width=None, draft=None):
        return (self.level if width is None else self.resolve(width))["id"]

    def tiers(self, scores=None):
        return [{"tier": l["id"], "size_pct": l["pct"], "eff_params_b": l["eff_params_b"], "mmlu_pt": l["mmlu_pt"],
                 "layers": self.n_layers - len(l["skip"]), "scores": (scores or {}).get(l["id"])}
                for l in self.levels]

    def logits(self, hidden):
        """Next-token scores from the last hidden states (the model's own head and soft cap)."""
        out = self.lm.model.embed_tokens.as_linear(hidden)
        cap = self.lm.final_logit_softcapping
        return logit_softcap(cap, out) if cap is not None else out

    # --- chat (used by the Cale server) -----------------------------------------------------
    max_context = MAX_CONTEXT
    supports = {"top_p": True, "seed": True, "stop": True, "guided": True}

    def memory_bandwidth(self):
        return memory_bandwidth()

    def interrupt(self):
        pass                                    # the prompt and answer loops check the Cancel object themselves

    def chat_stream(self, messages, max_tokens=256, width=None, draft=None, temperature=0.0, top_p=1.0, seed=None, stop=None, cancel=None, controller=None, guide=None, owner=None):
        """Multi-turn text chat. Yields {"delta": str} pieces, then {"done": stats}.
        messages: [{"role": "system"|"user"|"assistant", "content": str}].
        If the size needs FFN weights that are not in RAM, they are read from disk first and freed after the call."""
        lv = self.resolve(width) if width is not None else self.level
        read, read_s = self.ensure(lv)
        try:
            for ev in self._chat_stream(messages, max_tokens, lv, temperature, top_p, seed, stop, cancel, controller, guide, owner):
                if "done" in ev:
                    ev["done"].update(disk_mb=round(read / 1e6), disk_s=round(read_s, 3))
                yield ev
        except BaseException:                  # stopped early (client left, error, generator closed): leave nothing half-computed behind
            self.materialize()
            raise
        finally:
            self.settle()

    def _encode(self, messages):
        """The prompt as (token ids of shape (1, n), pixel values or None). A message with "parts" carries pictures: each picture
        becomes 256 tokens that stand for it. The tower that reads pictures is not loaded for text-only prompts."""
        tok = self.processor.tokenizer
        images, msgs = [], []
        for m in messages:
            if m.get("parts"):
                content = []
                for kind, v in m["parts"]:
                    if kind == "image":
                        content.append({"type": "image"})
                        images.append(v)
                    else:
                        content.append({"type": "text", "text": v})
            else:
                content = [{"type": "text", "text": m["content"]}]
            msgs.append({"role": m["role"], "content": content})
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if not images:
            return mx.array(tok.encode(text, add_special_tokens=False))[None], None
        from cale import media
        from mlx_vlm.utils import prepare_inputs
        got = prepare_inputs(self.processor, images=[media.load(b) for b in images], prompts=text)
        return got["input_ids"], got["pixel_values"]

    def prompt_ids(self, messages):
        """The prompt as token ids, shape (1, n). The relative-goal chooser needs n before it picks a size."""
        return self._encode(messages)[0]

    def _picture_tower(self):
        """Load the part of the model that reads pictures (about 0.3 GB). `_drop_picture_tower` frees it again."""
        if self.model.vision_tower is None:
            from mlx_vlm import load as vlm_load
            full, _ = vlm_load(self.path, lazy=True)
            self.model.vision_tower = full.vision_tower
            mx.eval(self.model.vision_tower.parameters())
            del full
        return self.model.vision_tower

    def _drop_picture_tower(self):
        self.model.vision_tower = None
        gc.collect()
        mx.clear_cache()

    def _prefill_pictures(self, ids, pixels, cache, cancel=None):
        """Like _prefill for a prompt that holds pictures: the picture tokens are replaced by what the picture tower reads."""
        self._picture_tower()
        try:
            emb = self.model.get_input_embeddings(input_ids=ids, pixel_values=pixels)
            mx.eval(emb.inputs_embeds, emb.per_layer_inputs)
        finally:
            self._drop_picture_tower()
        n, h, step = ids.shape[1], None, self.prefill_step()
        for i in range(0, n, step):
            if cancel:
                cancel.check()
            h = self.lm.model(None, inputs_embeds=emb.inputs_embeds[:, i:i + step], cache=cache,
                              per_layer_inputs=emb.per_layer_inputs[:, i:i + step])
            if i + step < n:
                mx.eval([c.state for c in cache])
        return h

    def materialize(self):
        """Finish every array the model still holds half-computed. MLX ties a half-computed array to the thread that made it,
        so a request that stops early (a client that left, an error) must call this before its thread ends. Otherwise the next
        request, on another thread, fails with "There is no Stream(gpu) in current thread" and the engine looks broken."""
        seen, arrays = set(), []

        def walk(o, depth=0):
            if id(o) in seen or depth > 8:
                return
            seen.add(id(o))
            if isinstance(o, mx.array):
                arrays.append(o)
            elif isinstance(o, dict):
                for v in list(o.values()):
                    walk(v, depth + 1)
            elif isinstance(o, (list, tuple, set)):
                for v in list(o):
                    walk(v, depth + 1)
            elif hasattr(o, "__dict__") and not isinstance(o, type):
                for v in list(vars(o).values()):
                    walk(v, depth + 1)

        walk(self.lm)
        walk(self)
        if arrays:
            mx.eval(arrays)
        return len(arrays)

    def prefill_step(self):
        window = getattr(self.lm.config, "sliding_window", None) or PREFILL_STEP
        return max(1, min(PREFILL_STEP, int(window)))

    def _prefill(self, ids, cache, cancel=None):
        """Read the prompt in pieces. Returns the hidden states of the last piece."""
        n, h, step = ids.shape[1], None, self.prefill_step()
        for i in range(0, n, step):
            if cancel:
                cancel.check()
            h = self.lm.model(ids[:, i:i + step], cache=cache)
            if i + step < n:
                mx.eval([c.state for c in cache])             # finish this piece before the next one starts
        return h

    def _sample(self, h, temperature, top_p):
        lg = self.logits(h[:, -1:])
        if temperature <= 0:
            return mx.argmax(lg, axis=-1)
        lg = lg / temperature
        if top_p >= 1.0:
            return mx.random.categorical(lg, axis=-1)
        probs = mx.softmax(lg, axis=-1)
        order = mx.argsort(-probs, axis=-1)                   # most likely first
        ps = mx.take_along_axis(probs, order, axis=-1)
        keep = (mx.cumsum(ps, axis=-1) - ps) < top_p          # the smallest set whose probability reaches top_p (always the first)
        pick_in_sorted = mx.random.categorical(mx.where(keep, mx.log(ps + 1e-20), -mx.inf), axis=-1)
        return mx.take_along_axis(order, pick_in_sorted[..., None], axis=-1).squeeze(-1)

    def _start_prompt(self, messages, cancel, owner=None):
        """Read the prompt. If it starts with exactly the tokens of the last prompt (a chat that grew by a turn, same size), the memory
        saved after that prompt is used and only the new tokens are read. Returns (ids, cache, hidden states of the last piece, tokens reused).
        The memory is saved right after the prompt, before any answer token: the next turn's prompt always starts with this prompt,
        while the answer may be written back slightly differently (the chat template trims blanks)."""
        ids, pixels = self._encode(messages)
        tokens = ids[0].tolist()
        pc = self.__dict__.get("_pc")
        cache, n = self.lm.make_cache(), 0
        if pixels is not None:                                       # a prompt with pictures is read whole and never saved for the next turn
            self._pc = None
            h = self._prefill_pictures(ids, pixels, cache, cancel)
            mx.eval(h, [c.state for c in cache])
            return ids, cache, h, 0
        if (owner is not None and pc and pc["owner"] == owner and pc["level"] == self.level["id"]      # never across apps: that would tell one app about another's prompts
                and len(pc["tokens"]) < len(tokens) and tokens[:len(pc["tokens"])] == pc["tokens"]):
            n = len(pc["tokens"])
            for c, (state, meta) in zip(cache, pc["snap"]):          # arrays are immutable: the saved ones are never changed by what comes next
                c.state, c.meta_state = state, meta
        h = self._prefill(ids[:, n:], cache, cancel)
        states = [c.state for c in cache]                            # each read of .state makes new lazy slices: finish exactly these
        mx.eval(h, states)                                           # arrays used from another thread next time must be finished
        self._pc = {"tokens": tokens, "snap": [(st, c.meta_state) for st, c in zip(states, cache)], "level": self.level["id"], "owner": owner}
        return ids, cache, h, n

    def forget_prompt(self):
        self._pc = None

    def _chat_stream(self, messages, max_tokens, width, temperature, top_p=1.0, seed=None, stop=None, cancel=None, controller=None, guide=None, owner=None):
        if guide is not None:
            yield from self._guided_stream(messages, max_tokens, width, temperature, top_p, seed, stop, cancel, controller, guide, owner)
            return
        sw = self.set(width)
        start_tier, switches = self.level["id"], []
        tok = self.processor.tokenizer
        mx.random.seed(secrets.randbits(32) if seed is None else int(seed))      # the random state is per thread: never rely on it
        stops = StopFilter(stop)
        t0 = time.perf_counter()
        ids, cache, h0, cached = self._start_prompt(messages, cancel, owner)

        nxt = self._sample(h0, temperature, top_p)
        mx.eval(nxt)
        t_first = time.perf_counter()
        out, sent, finish = [], "", None
        for _ in range(max_tokens):
            if cancel:
                cancel.check()
            v = int(nxt.item())
            if v in EOS:
                finish = "stop"
                break
            out.append(v)
            if controller and len(out) % STEP_TOKENS == 0:         # a size may change between two tokens: the cache stays
                new = controller(len(out), time.perf_counter() - t_first, self.level)
                if new is not None and new["id"] != self.level["id"]:
                    switches.append({"at_token": len(out), "from": self.level["id"], "to": new["id"]})
                    self.set(new)
            nxt = self._sample(self.lm.model(nxt.reshape(1, 1), cache=cache), temperature, top_p)
            mx.async_eval(nxt)
            s = tok.decode(out, skip_special_tokens=True)
            if not s.endswith("�") and len(s) > len(sent):
                text, hit = stops.feed(s[len(sent):])
                sent = s
                if text:
                    yield {"delta": text}
                if hit:
                    finish = "stop"
                    break
        else:
            finish = "length"
        if not stops.hit:
            s = tok.decode(out, skip_special_tokens=True)
            text, hit = stops.feed(s[len(sent):]) if len(s) > len(sent) else ("", False)
            tail = text + (stops.flush() if not hit else "")
            if tail:
                yield {"delta": tail}
            if hit:
                finish = "stop"
        end = time.perf_counter()
        yield {"done": {"tier": self.level["id"], "start_tier": start_tier, "switches": switches, "size_pct": self.level["pct"],
                        "switch_ms": round(sw, 1), "seconds": round(end - t0, 3), "first_token_s": round(t_first - t0, 3),
                        "decode_s": round(end - t_first, 3), "prompt_tokens": int(ids.shape[1]), "cached_tokens": cached,
                        "tokens": len(out), "finish": finish or "stop", "drafter": False}}

    def _piece_text(self, tid):
        """The text of a token for the guide, or None if it is special or not whole characters (then it is never allowed)."""
        import re
        cache = self.__dict__.setdefault("_pieces", {})
        if tid not in cache:
            tok = self.processor.tokenizer
            special = self.__dict__.setdefault("_special_ids", set(tok.all_special_ids))
            piece = None
            if tid not in special:
                raw = tok.convert_ids_to_tokens(int(tid))
                m = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", raw)
                if m:
                    b = int(m.group(1), 16)
                    piece = chr(b) if b < 0x80 else None            # a byte of a longer character: never allowed on its own
                else:
                    piece = raw.replace("\u2581", " ")
            cache[tid] = piece or None
        return cache[tid]

    def _candidates(self, h, k):
        """The k most likely next tokens, best first: [(id, log-probability)]."""
        lg = self.logits(h[:, -1:]).reshape(-1).astype(mx.float32)
        k = min(k, lg.shape[0] - 1)
        idx = mx.argpartition(-lg, kth=k, axis=-1)[:k]
        vals = lg[idx] - mx.logsumexp(lg)
        mx.eval(idx, vals)
        return sorted(zip(idx.tolist(), vals.tolist()), key=lambda x: -x[1])

    def _guided_stream(self, messages, max_tokens, width, temperature, top_p, seed, stop, cancel, controller, guide, owner=None):
        """Like _chat_stream, but every token must fit `guide` (cale/constrain.py): the answer is valid JSON that fits a schema."""
        import random
        from cale import constrain
        sw = self.set(width)
        start_tier, switches = self.level["id"], []
        rng = random.Random(secrets.randbits(32) if seed is None else int(seed))
        stops = StopFilter(stop)
        t0 = time.perf_counter()
        ids, cache, h, cached = self._start_prompt(messages, cancel, owner)
        mx.eval(h)
        t_first, n_out, finish = None, 0, None
        is_eog = lambda t: int(t) in EOS                                          # noqa: E731
        for _ in range(max_tokens):
            if cancel:
                cancel.check()
            pick = constrain.next_token(lambda k: self._candidates(h, k), guide, self._piece_text, is_eog, temperature, top_p, rng)
            tid, text = pick
            if t_first is None:
                t_first = time.perf_counter()
            if text is None:
                finish = "stop"
                break
            guide.feed(text)
            n_out += 1
            if controller and n_out % STEP_TOKENS == 0:
                new = controller(n_out, time.perf_counter() - t_first, self.level)
                if new is not None and new["id"] != self.level["id"]:
                    switches.append({"at_token": n_out, "from": self.level["id"], "to": new["id"]})
                    self.set(new)
            out_text, hit = stops.feed(text)
            if out_text:
                yield {"delta": out_text}
            if hit:
                finish = "stop"
                break
            if guide.finished():
                finish = "stop"
                break
            h = self.lm.model(mx.array([[tid]]), cache=cache)
        else:
            finish = "length"
        mx.eval(h, [c.state for c in cache])                       # leave nothing half-computed: arrays belong to the thread that made them
        if not stops.hit:
            tail = stops.flush()
            if tail:
                yield {"delta": tail}
        end = time.perf_counter()
        t_first = t_first or end
        yield {"done": {"tier": self.level["id"], "start_tier": start_tier, "switches": switches, "size_pct": self.level["pct"],
                        "switch_ms": round(sw, 1), "seconds": round(end - t0, 3), "first_token_s": round(t_first - t0, 3),
                        "decode_s": round(end - t_first, 3), "prompt_tokens": int(ids.shape[1]), "cached_tokens": cached,
                        "tokens": n_out, "finish": finish or "stop", "drafter": False}}

    def generate(self, prompt, max_tokens=256, width=None):
        """Greedy text for one user prompt (benchmarks). Returns {"text", "stats"}."""
        text, done = "", None
        for ev in self.chat_stream([{"role": "user", "content": prompt}], max_tokens, width):
            text += ev.get("delta", "")
            done = ev.get("done", done)
        return {"text": text, "stats": done}
