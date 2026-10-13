"""Gemma 3n nested sizes on llama.cpp (Windows and Linux). Same interface as nest.gemma3n.NestedGemma3n.

The model is one GGUF file: the stock file, or the same file after llama/prepare_gguf.py (down matrices in two halves).
A stock file is memory-mapped. A prepared file is repacked for speed and then lives in ordinary memory. A size is a list of layer widths given to the patched
llama.cpp (llama/patches). The FFN weights of a layer have three states, like the MLX model:
0 nothing in RAM, 1 the first half of the channels, 2 all channels.
"Reading from disk" means the OS reads the pages of the memory-mapped file.
"""
import codecs
import os
import random
import secrets
import time
from pathlib import Path

from nest.cancel import Cancelled
from nest.levels3n import BLOCK, MAX_CONTEXT, load_levels, pick
from nest.stops import StopFilter

from .. import constrain

from .. import oslayer
from . import devices
from .llamalib import Lib, devices as list_devices, memory_bandwidth

N_CTX = MAX_CONTEXT        # context size the library is loaded with (prompt plus answer, in tokens). Gemma 3n keeps 8 KB per token, so the full 32k costs 270 MB.
STEP_TOKENS = 8            # how often a size controller may change the size, in answer tokens
FIRST_SHARED = 20          # layers 20 and up share the KV cache of earlier layers, so they may be skipped
_FFN = ("gate", "up", "down")


def default_threads():
    if os.environ.get("CALE_THREADS"):
        return int(os.environ["CALE_THREADS"])
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except Exception:
        pass
    n = os.cpu_count() or 4
    return n // 2 if n >= 8 and n % 2 == 0 else n          # most computers with 8+ logical CPUs have two per core


def find_gguf(path):
    p = Path(path)
    if p.is_file():
        return p
    found = sorted(p.glob("*.gguf"))
    if not found:
        raise FileNotFoundError(f"no .gguf file in {p}")
    return found[0]


class LlamaNested:
    ple_on_disk = True                     # the per-layer embedding table is mapped: only the rows of the tokens used are read

    @classmethod
    def load(cls, path, width=1.0, ple_on_disk=True, threads=None, n_ctx=N_CTX, lib_path=None, **_):
        self = cls()
        self.path = str(find_gguf(path))
        self._kv_level = None                  # the size the key/value memory was made at, when it holds a conversation worth reusing
        self._kv_owner = None                  # the app that conversation belongs to: only that app may reuse it
        self.threads = threads or default_threads()
        self.lib_path = lib_path
        self.lib, self.placement = cls._open(self.path, self.threads, n_ctx, lib_path)
        self.n_layers = self.lib.n_layers
        self.levels = [l for l in load_levels(self.n_layers) if min(l["skip"], default=99) >= FIRST_SHARED]
        self.parts, self.layer_bytes, self.layer_read = [], [], []
        for i in range(self.n_layers):
            parts = {}
            for k in _FFN:
                r = self.lib.tensor_range(f"blk.{i}.ffn_{k}.weight")
                if r is None:
                    raise RuntimeError(f"weight blk.{i}.ffn_{k} not found")
                parts[k] = r
            g, u, d = (parts[k][1] for k in _FFN)
            self.parts.append(parts)
            # The first half of gate and up is one block of memory. The first half of down is one block only when the
            # file is in the prepared layout. In a stock file it is spread over every row, so down stays whole in state 1.
            self.layer_bytes.append([0, g // 2 + u // 2 + (d // 2 if self.lib.split else d), g + u + d])
            self.layer_read.append(g + u + d)
        # llama.cpp reads the whole file at load. The per-layer embedding table (over a GB) is only read one row per
        # token, so give its pages back: the OS reads the rows that are used, as the MLX model does with ple_on_disk.
        ple = self.lib.tensor_range("per_layer_token_embd.weight")
        if ple and ple_on_disk:
            self.lib.drop(*ple)
        if self.lib.repacked:
            self.lib.drop_file_mapping()       # the file pages that held the originals of the repacked weights
        # Repacked weights are all in memory after load. Mapped ones are read in now, so the first answers are not slow.
        self.state = [2] * self.n_layers if self.lib.repacked else [0] * self.n_layers    # what is in RAM now
        self.resident = [0] * self.n_layers    # what the holds ask for
        self.cap = 1.0
        self.draft = False
        self.level = None
        self.reside_states([2] * self.n_layers)
        self.set(width)
        self.generate("Hi", 2)
        return self

    @staticmethod
    def _open(path, threads, n_ctx, lib_path):
        """Load on the device the rule in devices.py picks. Any GPU problem falls back to the CPU: Cale must still start."""
        dev, layers, why = devices.choose(list_devices(lib_path), os.path.getsize(path))
        if dev is not None:
            try:
                lib = Lib(path, n_threads=threads, n_ctx=n_ctx, lib_path=lib_path, gpu=dev["handle"], n_gpu_layers=layers)
                return lib, f"{lib.placement}; chosen because {why}"
            except Exception as e:
                print(f"cale: the GPU did not load ({e}). Using the CPU.", flush=True)
                why = "the CPU (the GPU did not load)"
        lib = Lib(path, n_threads=threads, n_ctx=n_ctx, lib_path=lib_path)
        return lib, f"{lib.placement}; chosen because {why}"

    def close(self):
        if getattr(self, "lib", None):
            self.lib.close()
            self.lib = None

    # --- residency ---------------------------------------------------------------------------
    def need(self, lv):
        return [0 if i in lv["skip"] else lv["blocks"][i] for i in range(self.n_layers)]

    def _drop_tails(self, i, kinds=None):
        """The second half of this layer's FFN: gate and up rows, and down's second half when the file has it."""
        p = self.parts[i]
        for k in kinds or (("gate", "up") + (("down",) if self.lib.split else ())):
            a, n = p[k]
            self.lib.drop(a + n // 2, n - n // 2)

    def _drop(self, i, to):
        if to == 0:
            for k in _FFN:
                self.lib.drop(*self.parts[i][k])
        elif to == 1:
            self._drop_tails(i)
        self.state[i] = to

    def _read(self, i, to):
        p, cur = self.parts[i], self.state[i]
        restored = []
        for k in _FFN:
            # A repacked weight lives in ordinary memory, so dropped pages are empty: read the whole weight again from
            # the file. A weight mapped from the file comes back page by page, so ask for the pages this size needs.
            if self.lib.repacked and self.lib.restore(f"blk.{i}.ffn_{k}.weight"):
                restored.append(k)
                continue
            a, n = p[k]
            if k != "down" or self.lib.split:
                lo, hi = (0, n // 2) if to == 1 else (0 if cur == 0 else n // 2, n)
            else:
                lo, hi = (0, n) if cur == 0 else (0, 0)
            if hi > lo:
                self.lib.prefetch(a + lo, hi - lo)
        if to == 1 and restored:
            self._drop_tails(i, [k for k in restored if k != "down" or self.lib.split])
        self.state[i] = to

    def _read_back(self, todo):
        if not todo:
            return 0, 0.0
        t, read = time.perf_counter(), 0
        for i, n in todo:
            read += self.layer_bytes[i][n] - self.layer_bytes[i][self.state[i]]
            self._read(i, n)
        return read, time.perf_counter() - t

    def reside(self, levels):
        """Keep in RAM exactly what the union of `levels` needs. Frees the rest. Returns bytes freed."""
        return self.reside_states([max(col) for col in zip(*[self.need(self.resolve(l)) for l in levels])],
                                  first=self.resolve(levels[0]))

    def reside_states(self, states, first=None):
        before = self.ffn_bytes()
        self.resident = list(states)
        for i, st in enumerate(self.resident):
            if st < self.state[i]:
                self._drop(i, st)
        self._read_back([(i, st) for i, st in enumerate(self.resident) if st > self.state[i]])
        if first is not None:
            self.set(first)
        return before - self.ffn_bytes()

    def missing(self, lv):
        return [(i, n) for i, n in enumerate(self.need(lv)) if n > self.state[i]]

    def ensure(self, lv):
        return self._read_back(self.missing(lv))

    def settle(self):
        for i, st in enumerate(self.resident):
            if self.state[i] > st:
                self._drop(i, st)

    def ffn_bytes(self):
        return sum(self.layer_bytes[i][s] for i, s in enumerate(self.state))

    def size_bytes(self, lv):
        return sum(self.layer_bytes[i][n] for i, n in enumerate(self.need(lv)))

    def resident_now(self):
        """Bytes of the FFN weights this process has in RAM right now, asked from the OS (lower than ffn_bytes if the OS took pages)."""
        return sum(self.lib.resident_bytes(*self.parts[i][k]) for i in range(self.n_layers) for k in _FFN)

    # --- sizes -------------------------------------------------------------------------------
    def resolve(self, width):
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
            if self.level is None or lv["id"] != self.level["id"]:
                full = all(lv["blocks"][i] == 2 for i in range(self.n_layers)) and not lv["skip"]
                self.lib.set_widths(None if full else
                                    [0 if i in lv["skip"] else lv["blocks"][i] * BLOCK for i in range(self.n_layers)])
            self.level = lv
        return (time.perf_counter() - t) * 1000

    def set_cap(self, width):
        self.cap = float(width)

    def tier_name(self, width=None, draft=None):
        return (self.level if width is None else self.resolve(width))["id"]

    def tiers(self, scores=None):
        return [{"tier": l["id"], "size_pct": l["pct"], "eff_params_b": l["eff_params_b"], "mmlu_pt": l["mmlu_pt"],
                 "layers": self.n_layers - len(l["skip"]), "scores": (scores or {}).get(l["id"])}
                for l in self.levels]

    # --- what the engine and the profile ask every backend ---------------------------------------------
    def active_bytes(self):
        return oslayer.process_rss()

    def memory_bandwidth(self):
        """Bytes per second this computer reads from memory with this engine's threads. About 0.15 s. Answer speed is bound by it."""
        return memory_bandwidth(self.threads, self.lib_path)

    def clear_cache(self):
        self._kv_level = None
        self.lib.forget()

    def all_full(self):
        return all(s == 2 for s in self.state)

    def model_files(self):
        return [Path(self.path)]

    def prompt_ids(self, messages):
        return self.lib.tokenize(self.lib.chat_text(messages))

    def prompt_len(self, messages):
        return len(self.prompt_ids(messages))

    # --- chat --------------------------------------------------------------------------------
    supports = {"top_p": True, "seed": True, "stop": True, "guided": True}

    @property
    def max_context(self):
        return self.lib.n_ctx

    def interrupt(self):
        """Stop the prompt or answer in progress. Safe to call from another thread."""
        self.lib.cancel()

    def chat_stream(self, messages, max_tokens=256, width=None, draft=None, temperature=0.0, top_p=1.0, seed=None, stop=None, cancel=None, controller=None, guide=None, owner=None):
        """Multi-turn text chat. Yields {"delta": str} pieces, then {"done": stats}.
        If the size needs FFN weights that are not in RAM, they are read first and freed after the call."""
        lv = self.resolve(width) if width is not None else self.level
        read, read_s = self.ensure(lv)
        try:
            for ev in self._chat_stream(messages, max_tokens, lv, temperature, top_p, seed, stop, cancel, controller, guide, owner):
                if "done" in ev:
                    ev["done"].update(disk_mb=round(read / 1e6), disk_s=round(read_s, 3))
                yield ev
        finally:
            self.settle()

    def _chat_stream(self, messages, max_tokens, width, temperature, top_p=1.0, seed=None, stop=None, cancel=None, controller=None, guide=None, owner=None):
        sw = self.set(width)
        start_tier, switches = self.level["id"], []
        ids = self.prompt_ids(messages)
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        stops = StopFilter(stop)
        t0 = time.perf_counter()
        if cancel:
            cancel.check()
        seed = secrets.randbits(32) if seed is None else int(seed) & 0xFFFFFFFF
        gstate = None
        if guide is not None:
            if not self.lib.has_guided:
                raise RuntimeError("this engine library cannot guide output: rebuild it with llama/build")
            gstate = {"guide": guide, "pending": None, "t": temperature, "p": top_p, "rng": random.Random(seed)}
        reuse = owner is not None and self._kv_owner == owner and self._kv_level == self.level["id"]      # never across apps: that would tell one app about another's prompts
        self._kv_level = None                                       # until this answer ends cleanly, the memory is not known to match
        self.lib.begin(ids, temperature, seed=seed, top_p=top_p, reuse=reuse)
        cached = self.lib.reused
        out, finish, t_first = 0, None, None
        for _ in range(max_tokens):
            tok = self._guided_next(gstate) if gstate else self.lib.next()
            if t_first is None:
                t_first = time.perf_counter()                       # the prompt is read and the first token chosen
            if tok == -1 and self.lib.error() == "cancelled":
                raise Cancelled()
            if cancel:
                cancel.check()
            if tok < 0 or self.lib.is_eog(tok):
                finish = "stop"
                break
            out += 1
            if controller and out % STEP_TOKENS == 0:               # a size may change between two tokens: the cache stays
                new = controller(out, time.perf_counter() - t_first, self.level)
                if new is not None and new["id"] != self.level["id"]:
                    switches.append({"at_token": out, "from": self.level["id"], "to": new["id"]})
                    self.set(new)
            s = dec.decode(self.lib.piece(tok))
            if s:
                text, hit = stops.feed(s)
                if text:
                    yield {"delta": text}
                if hit:
                    finish = "stop"
                    break
            if gstate and gstate["guide"].finished():              # the answer is a complete document: nothing more to say
                finish = "stop"
                break
        else:
            finish = "length"
        if not stops.hit:
            tail = stops.feed(dec.decode(b"", final=True))
            tail = tail[0] + (stops.flush() if not tail[1] else "")
            if tail:
                yield {"delta": tail}
        end = time.perf_counter()
        t_first = t_first or end
        if not switches:
            self._kv_level, self._kv_owner = self.level["id"], owner  # the memory holds this app's conversation at this size: its next turn can reuse it
        yield {"done": {"tier": self.level["id"], "start_tier": start_tier, "switches": switches, "size_pct": self.level["pct"],
                        "switch_ms": round(sw, 1), "seconds": round(end - t0, 3), "first_token_s": round(t_first - t0, 3),
                        "decode_s": round(end - t_first, 3), "prompt_tokens": len(ids), "cached_tokens": cached,
                        "tokens": out, "finish": finish or "stop", "drafter": False}}

    def _text(self, tok):
        """The text of a token for the guide, or None if it is a special token or not whole characters (then it is never allowed)."""
        cache = self.__dict__.setdefault("_texts", {})
        if tok not in cache:
            try:
                cache[tok] = self.lib.piece(tok).decode("utf-8") or None
            except UnicodeDecodeError:
                cache[tok] = None
        return cache[tok]

    def _guided_next(self, g):
        """The next token of a guided answer: the best of the model's candidates that the guide accepts. Reads the last token in first."""
        if g["pending"] is not None:
            self.lib.accept(g["pending"])
            g["pending"] = None
        pick = constrain.next_token(self.lib.topk, g["guide"], self._text, self.lib.is_eog, g["t"], g["p"], g["rng"])
        tid, text = pick
        if text is not None:
            g["guide"].feed(text)
            g["pending"] = tid
        return tid

    def generate(self, prompt, max_tokens=256, width=None):
        text, done = "", None
        for ev in self.chat_stream([{"role": "user", "content": prompt}], max_tokens, width):
            text += ev.get("delta", "")
            done = ev.get("done", done)
        return {"text": text, "stats": done}
