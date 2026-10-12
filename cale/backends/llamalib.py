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


class Lib:
    def __init__(self, model_path, n_threads=4, n_ctx=8192, lib_path=None, repack=0):
        self.c = ctypes.CDLL(find_library(lib_path))
        c = self.c
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
        u64 = ctypes.c_uint64
        c.cale_tensor_range.restype, c.cale_tensor_range.argtypes = i32, [vp, ctypes.c_char_p, ctypes.POINTER(u64), ctypes.POINTER(u64)]
        c.cale_pages.restype, c.cale_pages.argtypes = u64, [i32, u64, u64]
        self.h = c.cale_load(os.fsencode(str(model_path)), n_threads, n_ctx, repack)
        if not self.h:
            raise RuntimeError(c.cale_last_error().decode())
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

    def begin(self, tokens, temperature=0.0, seed=1234, top_p=1.0):
        arr = (ctypes.c_int32 * len(tokens))(*tokens)
        if self.c.cale_begin2(self.h, arr, len(tokens), temperature, top_p, seed) != 0:
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

    def drop(self, addr, nbytes):
        return self.c.cale_pages(0, addr, nbytes)

    def prefetch(self, addr, nbytes):
        return self.c.cale_pages(1, addr, nbytes)

    def resident_bytes(self, addr, nbytes):
        return self.c.cale_pages(2, addr, nbytes)

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
