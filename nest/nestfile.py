"""The .nest file: every precision tier and width of the 12B in one file.

Layout:
    8 bytes  magic b"NEST0001"
    8 bytes  header length (little-endian u64)
    header   JSON
    data     starts at the first 16 KiB boundary after the header

Every tensor starts on a 16 KiB boundary (the Apple Silicon page size). Offsets
in the header are relative to the data start. Tensors are stored in region order:

| Region | Holds | Needed by |
|---|---|---|
| X  | config, tokenizer and chat template files (raw bytes) | all tiers |
| A  | plane A of every table (QAT 4-bit, exactly as in models/qg4) | all tiers |
| N4 | QAT norms and other unquantized tensors | 4-bit tier |
| B  | row_scale and plane B of every table | 8- and 16-bit tiers |
| N8 | original norms and other unquantized tensors | 8- and 16-bit tiers |
| C  | plane C of every table | 16-bit tier |

So the 4-bit tier is a prefix of the file (X, A, N4).

Tensor names: "<module path>|<plane>|<part>", e.g.
    language_model.model.layers.3.mlp.down_proj|B1|weight   (plane B, column block 1)
    language_model.model.layers.3.mlp.gate_proj|A|scales
    language_model.model.layers.3.mlp.gate_proj|B|row_scale
    language_model.model.layers.3.input_layernorm.weight|N8
"""
import json
import mmap
import os
import struct

import mlx.core as mx
import numpy as np

MAGIC = b"NEST0001"
ALIGN = 16384
REGIONS = ("X", "A", "N4", "B", "N8", "C")
_NP = {"uint32": np.uint32, "bfloat16": np.uint16, "float32": np.float32, "uint8": np.uint8,
       "float16": np.float16}


def _align(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def _dtype_name(a):
    return str(a.dtype).split(".")[-1]


def _to_numpy(a):
    if a.dtype == mx.bfloat16:
        return np.array(a.view(mx.uint16))
    return np.array(a)


class NestWriter:
    """Lay out every tensor first (sizes come from shapes), then fill them in any order."""

    def __init__(self, path, specs, meta):
        """specs: list of (name, region, dtype_name, shape), any order (sorted here by region,
        then by list order). meta: extra header fields (config, tables)."""
        order = {r: i for i, r in enumerate(REGIONS)}
        specs = sorted(enumerate(specs), key=lambda e: (order[e[1][1]], e[0]))
        self.tensors, off, regions = {}, 0, {}
        for _, (name, region, dt, shape) in specs:
            nbytes = int(np.prod(shape)) * np.dtype(_NP[dt]).itemsize
            self.tensors[name] = {"region": region, "dtype": dt, "shape": list(shape),
                                  "offset": off, "nbytes": nbytes}
            lo, hi = regions.get(region, (off, off))
            regions[region] = (min(lo, off), off + nbytes)
            off = _align(off + nbytes)
        header = dict(meta, format="nest-v1", align=ALIGN, regions=regions, tensors=self.tensors)
        hb = json.dumps(header).encode()
        self.data_start = _align(16 + len(hb))
        self.size = self.data_start + off
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.write(self.fd, MAGIC + struct.pack("<Q", len(hb)) + hb)
        os.ftruncate(self.fd, self.size)
        self.written = {}

    def write(self, name, a, row=0):
        """Write array a (or a block of rows starting at `row`) into tensor `name`."""
        t = self.tensors[name]
        if isinstance(a, (bytes, bytearray)):
            buf = bytes(a)
        else:
            assert _dtype_name(a) == t["dtype"], (name, a.dtype, t["dtype"])
            assert list(a.shape[1:]) == t["shape"][1:], (name, a.shape, t["shape"])
            buf = np.ascontiguousarray(_to_numpy(a)).tobytes()
        row_bytes = t["nbytes"] // t["shape"][0] if t["shape"] else t["nbytes"]
        start = row * row_bytes
        assert start + len(buf) <= t["nbytes"], name
        os.pwrite(self.fd, buf, self.data_start + t["offset"] + start)
        self.written[name] = self.written.get(name, 0) + len(buf)

    def close(self):
        missing = [n for n, t in self.tensors.items() if self.written.get(n, 0) != t["nbytes"]]
        os.close(self.fd)
        assert not missing, f"{len(missing)} tensors not fully written, e.g. {missing[:3]}"


class NestFile:
    def __init__(self, path):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        pre = os.pread(self.fd, 16, 0)
        assert pre[:8] == MAGIC, "not a nest file"
        (hl,) = struct.unpack("<Q", pre[8:16])
        self.header = json.loads(os.pread(self.fd, hl, 16))
        self.tensors = self.header["tensors"]
        self.data_start = _align(16 + hl)
        self.mm = mmap.mmap(self.fd, 0, prot=mmap.PROT_READ)

    def _view(self, name, rows=None):
        t = self.tensors[name]
        shape = list(t["shape"])
        nbytes = t["nbytes"]
        if rows is not None and rows < shape[0]:
            nbytes = nbytes // shape[0] * rows
            shape[0] = rows
        start = self.data_start + t["offset"]
        return t, shape, start, nbytes

    def _wrap(self, t, shape, raw):
        a = np.frombuffer(raw, dtype=_NP[t["dtype"]]).reshape(shape)
        m = mx.array(a)
        return m.view(mx.bfloat16) if t["dtype"] == "bfloat16" else m

    def array(self, name, rows=None):
        """Copy a tensor (or its first `rows` rows) from the page cache into an MLX array."""
        t, shape, start, nbytes = self._view(name, rows)
        return self._wrap(t, shape, memoryview(self.mm)[start:start + nbytes])

    def read(self, name, rows=None, fd=None):
        """Read with preadv on `fd` (e.g. an F_NOCACHE descriptor). Returns (array, bytes read).

        The buffer is page-aligned (an anonymous mapping): macOS bypasses the page cache
        for F_NOCACHE reads only when the buffer is aligned. Unaligned reads were cached
        and ~3.5x slower (1.6 vs 5.7 GB/s)."""
        t, shape, start, nbytes = self._view(name, rows)
        buf = mmap.mmap(-1, _align(nbytes))
        got = os.preadv(fd if fd is not None else self.fd, [buf], start)
        assert got >= nbytes, (name, got, nbytes)
        a = self._wrap(t, shape, memoryview(buf)[:nbytes])
        mx.eval(a)
        buf.close()
        return a, nbytes

    def blob(self, name):
        t = self.tensors[name]
        start = self.data_start + t["offset"]
        return bytes(self.mm[start:start + t["nbytes"]])

    def region_bytes(self, region):
        return sum(t["nbytes"] for t in self.tensors.values() if t["region"] == region)
