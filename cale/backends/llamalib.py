"""ctypes bridge to libcale_llama (the patched llama.cpp plus llama/shim/cale_llama.cpp).

Standard library only. One `Lib` per loaded model. The calls are the C functions in the shim, one to one.
"""
import ctypes
import os
import sys
from pathlib import Path

from nest.cancel import Cancelled

NAME = {"win32": "cale_llama.dll", "darwin": "libcale_llama.dylib"}.get(sys.platform, "libcale_llama.so")


def find_library(extra=None):
    """The shared library: $CALE_LLAMA_LIB, next to this file, or in Cale's home (lib/)."""
    here = Path(__file__).resolve().parent
    cands = [os.environ.get("CALE_LLAMA_LIB"), extra, here / NAME, here.parent / "lib" / NAME]
    try:
        from .. import paths
        cands.append(paths.HOME / "lib" / NAME)
    except Exception:
        pass
    for c in cands:
        if c and Path(c).is_file():
            return str(c)
    raise FileNotFoundError(f"{NAME} not found. Build it with llama/build.sh, or set CALE_LLAMA_LIB.")


def devices(lib_path=None):
    """Compute devices other than the CPU, as dicts: handle, type (GPU, IGPU, ACCEL), name, description, free, total (bytes).
    Empty when the library has no GPU backend. The handle is only valid in this process. Choose a device by its type,
    name and memory, never by its number: the order changes when a driver changes."""
    c = ctypes.CDLL(find_library(lib_path))
    if not hasattr(c, "cale_devices"):
        return []
    buf = ctypes.create_string_buffer(8192)
    c.cale_devices.restype, c.cale_devices.argtypes = ctypes.c_int32, [ctypes.c_char_p, ctypes.c_int32]
    if c.cale_devices(buf, 8192) < 0:
        return []
    out = []
    for line in buf.value.decode().splitlines():
        h, kind, name, desc, free, total = line.split("|")
        out.append({"handle": int(h), "type": kind, "name": name, "description": desc, "free": int(free), "total": int(total)})
    return out


def readback(lib_path=None):
    """True if this library can let a GPU read prompts while the weights stay repacked in RAM (option D, llama/patches/0002)."""
    c = ctypes.CDLL(find_library(lib_path))
    if not hasattr(c, "cale_readback"):
        return False
    c.cale_readback.restype, c.cale_readback.argtypes = ctypes.c_int32, []
    return bool(c.cale_readback())


def offload_min_batch(gpu, lib_path=None):
    """The fewest prompt tokens in one pass at which the GPU backend takes a matrix multiplication from the CPU. 0 = never, or unknown."""
    c = ctypes.CDLL(find_library(lib_path))
    if gpu is None or not hasattr(c, "cale_offload_min_batch"):
        return 0
    c.cale_offload_min_batch.restype, c.cale_offload_min_batch.argtypes = ctypes.c_int32, [ctypes.c_int32]
    return int(c.cale_offload_min_batch(gpu))


def memory_bandwidth(threads, lib_path=None, mb_per_thread=48):
    """Bytes per second this computer reads from memory with `threads` threads. About 0.15 s. No model needed. 0.0 if unknown."""
    c = ctypes.CDLL(find_library(lib_path))
    if not hasattr(c, "cale_membw"):
        return 0.0
    c.cale_membw.restype, c.cale_membw.argtypes = ctypes.c_double, [ctypes.c_int32, ctypes.c_int32]
    return max(c.cale_membw(threads, mb_per_thread), c.cale_membw(threads, mb_per_thread))


class Lib:
    def __init__(self, model_path, n_threads=4, n_ctx=8192, lib_path=None, repack=0,
                 gpu=None, n_gpu_layers=-1, no_host=True, fit_margin_mb=300):
        """gpu: None = CPU only, else a device handle from devices(). n_gpu_layers -1 = llama.cpp's own fit chooses.
        no_host: keep the weights that stay on the CPU repackable (see the shim). Always right unless a developer asks otherwise."""
        self.c = ctypes.CDLL(find_library(lib_path))
        c = self.c
        self.gpu, self.lib_path = gpu, lib_path
        vp, i32, i32p = ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)
        cp = ctypes.POINTER(ctypes.c_char_p)
        c.cale_last_error.restype = ctypes.c_char_p
        c.cale_load.restype, c.cale_load.argtypes = vp, [ctypes.c_char_p, i32, i32, i32]
        c.cale_repacked.restype, c.cale_repacked.argtypes = i32, [vp]
        c.cale_split.restype, c.cale_split.argtypes = i32, [vp]
        c.cale_drop_file_mapping.restype, c.cale_drop_file_mapping.argtypes = ctypes.c_uint64, [vp]
        c.cale_restore.restype, c.cale_restore.argtypes = i32, [vp, ctypes.c_char_p]
        c.cale_free.argtypes = [vp]
        c.cale_n_layers.restype, c.cale_n_layers.argtypes = i32, [vp]
        c.cale_n_ctx.restype, c.cale_n_ctx.argtypes = i32, [vp]
        c.cale_set_threads.argtypes = [vp, i32]
        c.cale_set_widths.restype, c.cale_set_widths.argtypes = i32, [vp, i32p, i32]
        c.cale_chat_text.restype, c.cale_chat_text.argtypes = i32, [vp, cp, cp, i32, ctypes.c_char_p, i32]
        c.cale_tokenize.restype, c.cale_tokenize.argtypes = i32, [vp, ctypes.c_char_p, i32, i32p, i32]
        c.cale_piece.restype, c.cale_piece.argtypes = i32, [vp, i32, ctypes.c_char_p, i32]
        c.cale_is_eog.restype, c.cale_is_eog.argtypes = i32, [vp, i32]
        c.cale_begin.restype, c.cale_begin.argtypes = i32, [vp, i32p, i32, ctypes.c_float, ctypes.c_uint32]
        c.cale_cancel.restype, c.cale_cancel.argtypes = None, [vp]
        c.cale_begin2.restype, c.cale_begin2.argtypes = i32, [vp, i32p, i32, ctypes.c_float, ctypes.c_float, ctypes.c_uint32]
        c.cale_next.restype, c.cale_next.argtypes = i32, [vp]
        c.cale_nll.restype, c.cale_nll.argtypes = ctypes.c_double, [vp, i32p, i32, i32]
        self.has_reuse = hasattr(c, "cale_begin3")
        if self.has_reuse:
            c.cale_begin3.restype, c.cale_begin3.argtypes = i32, [vp, i32p, i32, ctypes.c_float, ctypes.c_float, ctypes.c_uint32, i32]
            c.cale_reused.restype, c.cale_reused.argtypes = i32, [vp]
            c.cale_forget.restype, c.cale_forget.argtypes = None, [vp]
        self.has_slots = hasattr(c, "cale_batch_step")
        self.has_logprobs = False
        if self.has_slots:
            c.cale_slots.restype, c.cale_slots.argtypes = i32, [vp]
            c.cale_slot_open.restype, c.cale_slot_open.argtypes = i32, [vp, i32, i32p, i32, ctypes.c_float, ctypes.c_float, ctypes.c_uint32, i32, i32]
            c.cale_slot_close.restype, c.cale_slot_close.argtypes = None, [vp, i32]
            c.cale_slot_forget.restype, c.cale_slot_forget.argtypes = None, [vp, i32]
            c.cale_batch_step.restype, c.cale_batch_step.argtypes = i32, [vp, i32, i32p, i32p, i32p, i32]
            c.cale_slot_topk.restype, c.cale_slot_topk.argtypes = i32, [vp, i32, i32, i32p, ctypes.POINTER(ctypes.c_float)]
            c.cale_slot_accept.restype, c.cale_slot_accept.argtypes = i32, [vp, i32, i32]
            self.has_logprobs = hasattr(c, "cale_slot_logprobs")
            if self.has_logprobs:
                c.cale_slot_logprobs.restype = i32
                c.cale_slot_logprobs.argtypes = [vp, i32, i32, i32, i32p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
            c.cale_slot_reused.restype, c.cale_slot_reused.argtypes = i32, [vp, i32]
            c.cale_slot_common.restype, c.cale_slot_common.argtypes = i32, [vp, i32, i32p, i32]
            c.cale_slot_tokens.restype, c.cale_slot_tokens.argtypes = i32, [vp, i32]
        self.has_guided = hasattr(c, "cale_topk")
        if self.has_guided:
            c.cale_topk.restype, c.cale_topk.argtypes = i32, [vp, i32, i32p, ctypes.POINTER(ctypes.c_float)]
            c.cale_accept.restype, c.cale_accept.argtypes = i32, [vp, i32]
        u64 = ctypes.c_uint64
        c.cale_tensor_range.restype, c.cale_tensor_range.argtypes = i32, [vp, ctypes.c_char_p, ctypes.POINTER(u64), ctypes.POINTER(u64)]
        c.cale_pages.restype, c.cale_pages.argtypes = u64, [i32, u64, u64]
        if hasattr(c, "cale_set_slots"):               # not through the environment: a Windows library does not see variables set after the process started
            c.cale_set_slots.restype, c.cale_set_slots.argtypes = None, [i32]
            c.cale_set_slots(int(os.environ.get("CALE_SLOTS", "1") or 1))
        if hasattr(c, "cale_load_dev"):
            c.cale_load_dev.restype, c.cale_load_dev.argtypes = vp, [ctypes.c_char_p, i32, i32, i32, i32, i32, i32, i32]
            c.cale_placement.restype, c.cale_placement.argtypes = i32, [vp, ctypes.c_char_p, i32]
            self.h = c.cale_load_dev(os.fsencode(str(model_path)), n_threads, n_ctx, repack,
                                     -1 if gpu is None else gpu, n_gpu_layers, 1 if no_host else 0, fit_margin_mb)
        else:
            if gpu is not None:
                raise RuntimeError("this library has no GPU backend")
            self.h = c.cale_load(os.fsencode(str(model_path)), n_threads, n_ctx, repack)
        if not self.h:
            raise RuntimeError(c.cale_last_error().decode())
        self.placement = "device=cpu"
        if hasattr(c, "cale_placement"):
            pbuf = ctypes.create_string_buffer(512)
            if c.cale_placement(self.h, pbuf, 512) >= 0:
                self.placement = pbuf.value.decode()
        self.repacked = bool(c.cale_repacked(self.h))
        self.split = bool(c.cale_split(self.h))      # the file has each down matrix as two halves (llama/prepare_gguf.py)
        self.n_layers = c.cale_n_layers(self.h)
        self.n_ctx = c.cale_n_ctx(self.h)

    def close(self):
        if self.h:
            self.c.cale_free(self.h)
            self.h = None

    def error(self):
        return self.c.cale_last_error().decode()

    def set_widths(self, widths):
        """widths: a list with one number per layer (0 skips the layer, N uses N FFN channels, -1 full), or None for full."""
        if not widths:
            return self.c.cale_set_widths(self.h, None, 0)
        arr = (ctypes.c_int32 * len(widths))(*widths)
        r = self.c.cale_set_widths(self.h, arr, len(widths))
        if r != 0:
            raise RuntimeError(self.error())
        return r

    def chat_text(self, messages):
        n = len(messages)
        roles = (ctypes.c_char_p * n)(*[m["role"].encode() for m in messages])
        conts = (ctypes.c_char_p * n)(*[m["content"].encode() for m in messages])
        cap = 4 * sum(len(m["content"]) + 32 for m in messages) + 256
        buf = ctypes.create_string_buffer(cap)
        r = self.c.cale_chat_text(self.h, roles, conts, n, buf, cap)
        if r < 0:
            raise RuntimeError(self.error())
        if r > cap:
            buf = ctypes.create_string_buffer(r + 1)
            r = self.c.cale_chat_text(self.h, roles, conts, n, buf, r + 1)
        return buf.raw[:r].decode("utf-8", "replace")

    def tokenize(self, text, add_special=True):
        raw = text.encode()
        cap = len(raw) + 16
        arr = (ctypes.c_int32 * cap)()
        n = self.c.cale_tokenize(self.h, raw, 1 if add_special else 0, arr, cap)
        if n < 0:
            cap = -n
            arr = (ctypes.c_int32 * cap)()
            n = self.c.cale_tokenize(self.h, raw, 1 if add_special else 0, arr, cap)
        return list(arr[:n])

    def piece(self, tok):
        buf = ctypes.create_string_buffer(256)
        n = self.c.cale_piece(self.h, tok, buf, 256)
        return buf.raw[:max(n, 0)]

    def is_eog(self, tok):
        return bool(self.c.cale_is_eog(self.h, tok))

    def begin(self, tokens, temperature=0.0, seed=1234, top_p=1.0, reuse=False):
        """Read the prompt. reuse=True keeps the start of the memory that matches (only if it was made at the same size)."""
        arr = (ctypes.c_int32 * len(tokens))(*tokens)
        if self.has_reuse:
            r = self.c.cale_begin3(self.h, arr, len(tokens), temperature, top_p, seed, 1 if reuse else 0)
        else:
            r = self.c.cale_begin2(self.h, arr, len(tokens), temperature, top_p, seed)
        if r != 0:
            if self.error() == "cancelled":
                raise Cancelled()
            raise RuntimeError(self.error())

    # --- answers that run together (see llama/shim): one slot per answer ---
    @property
    def n_slots(self):
        return self.c.cale_slots(self.h) if self.has_slots else 1

    def slot_open(self, slot, tokens, temperature=0.0, top_p=1.0, seed=1234, guided=False, reuse=False):
        """Start an answer in a free slot. Returns how many prompt tokens were kept from the slot's memory."""
        arr = (ctypes.c_int32 * len(tokens))(*tokens)
        r = self.c.cale_slot_open(self.h, slot, arr, len(tokens), temperature, top_p, seed, 1 if guided else 0, 1 if reuse else 0)
        if r < 0:
            raise RuntimeError(self.error())
        return r

    def slot_common(self, slot, tokens):
        """How many tokens from the start of `tokens` this slot's memory already holds."""
        arr = (ctypes.c_int32 * len(tokens))(*tokens)
        return self.c.cale_slot_common(self.h, slot, arr, len(tokens))

    def slot_tokens(self, slot):
        """How many tokens of key/value memory this slot holds. All slots share one pool of n_ctx tokens."""
        return self.c.cale_slot_tokens(self.h, slot)

    def slot_close(self, slot):
        self.c.cale_slot_close(self.h, slot)

    def slot_forget(self, slot):
        self.c.cale_slot_forget(self.h, slot)

    def batch_step(self, prefill_budget=64):
        """One step for all running slots. Returns [(slot, kind, token)]: kind 1 = a token was sampled, 2 = a guided slot waits for slot_topk/slot_accept."""
        n = max(self.n_slots, 1)
        sl, kd, tk = (ctypes.c_int32 * n)(), (ctypes.c_int32 * n)(), (ctypes.c_int32 * n)()
        r = self.c.cale_batch_step(self.h, prefill_budget, sl, kd, tk, n)
        if r < 0:
            if self.error() == "cancelled":
                raise Cancelled()
            raise RuntimeError(self.error())
        return [(sl[i], kd[i], tk[i]) for i in range(r)]

    def slot_topk(self, slot, k):
        ids, lp = (ctypes.c_int32 * k)(), (ctypes.c_float * k)()
        n = self.c.cale_slot_topk(self.h, slot, k, ids, lp)
        if n < 0:
            raise RuntimeError(self.error())
        return [(ids[i], lp[i]) for i in range(n)]

    @property
    def n_batch(self):
        """Most tokens the engine reads in one pass."""
        if not hasattr(self.c, "cale_n_batch"):
            return 512
        self.c.cale_n_batch.restype, self.c.cale_n_batch.argtypes = ctypes.c_int32, [ctypes.c_void_p]
        return int(self.c.cale_n_batch(self.h))

    def slot_logprobs(self, slot, tok, k):
        """For the token a slot just got: (its log-probability, [(token, log-probability) of the k most likely tokens])."""
        ids, lp, chosen = (ctypes.c_int32 * max(k, 1))(), (ctypes.c_float * max(k, 1))(), ctypes.c_float()
        n = self.c.cale_slot_logprobs(self.h, slot, tok, k, ids, lp, ctypes.byref(chosen))
        if n < 0:
            raise RuntimeError(self.error())
        return chosen.value, [(ids[i], lp[i]) for i in range(n)]

    def slot_accept(self, slot, tok):
        if self.c.cale_slot_accept(self.h, slot, tok) != 0:
            raise RuntimeError(self.error())

    def slot_reused(self, slot):
        return self.c.cale_slot_reused(self.h, slot)

    @property
    def reused(self):
        """How many tokens of the last prompt were taken from the memory instead of being read again."""
        return self.c.cale_reused(self.h) if self.has_reuse else 0

    def forget(self):
        if self.has_reuse:
            self.c.cale_forget(self.h)

    def topk(self, k):
        """The k most likely next tokens, best first: [(token id, log-probability)]. For guided decoding."""
        ids, lp = (ctypes.c_int32 * k)(), (ctypes.c_float * k)()
        n = self.c.cale_topk(self.h, k, ids, lp)
        if n < 0:
            raise RuntimeError(self.error())
        return [(ids[i], lp[i]) for i in range(n)]

    def accept(self, tok):
        """Read a token chosen by the caller into the context."""
        r = self.c.cale_accept(self.h, tok)
        if r != 0:
            if self.error() == "cancelled":
                raise Cancelled()
            raise RuntimeError(self.error())

    def cancel(self):
        """Stop the work in progress. Safe to call from another thread."""
        self.c.cale_cancel(self.h)

    def next(self):
        return self.c.cale_next(self.h)

    def nll(self, tokens, first=1):
        """Mean negative log-likelihood (nats) of tokens[first:], each given the tokens before it."""
        arr = (ctypes.c_int32 * len(tokens))(*tokens)
        v = self.c.cale_nll(self.h, arr, len(tokens), first)
        if v < 0:
            raise RuntimeError(self.error())
        return v

    def generate_ids(self, tokens, max_tokens, temperature=0.0):
        """Token ids of an answer, stopping at the end-of-answer token."""
        self.begin(tokens, temperature)
        out = []
        for _ in range(max_tokens):
            t = self.next()
            if t < 0 or self.is_eog(t):
                break
            out.append(t)
        return out

    # --- where the weights live in memory (the OS decides what stays in RAM; these calls nudge it) ---
    def tensor_range(self, name):
        """(address, bytes) of a named weight, or None."""
        a, n = ctypes.c_uint64(0), ctypes.c_uint64(0)
        if self.c.cale_tensor_range(self.h, name.encode(), ctypes.byref(a), ctypes.byref(n)) != 0:
            return None
        return a.value, n.value

    # Address 0 means the weight is on a GPU: there are no pages of ours to move.
    def drop(self, addr, nbytes):
        return self.c.cale_pages(0, addr, nbytes) if addr else 0

    def prefetch(self, addr, nbytes):
        return self.c.cale_pages(1, addr, nbytes) if addr else 0

    def resident_bytes(self, addr, nbytes):
        return self.c.cale_pages(2, addr, nbytes) if addr else 0

    def restore(self, name):
        """Read a weight from the file into memory again. Needed after drop() only for repacked weights.
        Returns True if it was read, False if the weight is mapped from the file and needs nothing."""
        r = self.c.cale_restore(self.h, name.encode())
        if r < 0:
            raise RuntimeError(self.error())
        return r == 0

    def drop_file_mapping(self):
        """Give back every page of the model file that this process has mapped. Returns bytes. Safe at any time."""
        return self.c.cale_drop_file_mapping(self.h)
