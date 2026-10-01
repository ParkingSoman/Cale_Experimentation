"""Which plane blocks live in RAM, and streaming of the rest from SSD.

| Part | Job |
|---|---|
| priority()   | order of tensors for residency: small tensors, plane A, B, C; narrow-width blocks before width tails |
| TierStore    | pread with F_NOCACHE (reads bypass the page cache); counts bytes and wait time |
| Prefetcher   | one reader thread; reads layer i+1's streamed blocks while layer i runs |
"""
import fcntl
import os
import threading
import time

import mlx.core as mx

F_NOCACHE = 48   # macOS fcntl: do not keep this descriptor's reads in the page cache


def table_of(name):
    return name.split("|")[0]


def plane_of(name):
    p = name.split("|")
    return p[1][0] if len(p) > 2 else p[1]


def block_of(name):
    tag = name.split("|")[1]
    return int(tag[1:]) if len(tag) > 1 and tag[1:].isdigit() else 0


def priority(header):
    """Tensor names, most important to keep in RAM first."""
    tables = header["tables"]

    def key(name):
        t = header["tensors"][name]
        region = t["region"]
        if region in ("X", "N4", "N8"):
            return (0, 0, 0, 0)
        tab = tables[table_of(name)]
        plane = {"A": 1, "B": 2, "C": 3}[region]
        tail = 1 if block_of(name) > 0 else 0          # down_proj width tails after heads
        return (plane, tail, tab["layer"], t["offset"])

    return sorted(header["tensors"], key=key)


class TierStore:
    """Uncached reads from the nest file, with counters."""

    def __init__(self, nf):
        self.nf = nf
        self.fd = os.open(nf.path, os.O_RDONLY)
        fcntl.fcntl(self.fd, F_NOCACHE, 1)
        self.bytes = 0
        self.read_s = 0.0
        self.lock = threading.Lock()

    def read(self, name, rows=None):
        t = time.perf_counter()
        a, n = self.nf.read(name, rows, fd=self.fd)
        with self.lock:
            self.bytes += n
            self.read_s += time.perf_counter() - t
        return a


class Prefetcher:
    """Reads the streamed tensors of one layer ahead, on one thread."""

    def __init__(self, store):
        self.store = store
        self.jobs = {}          # layer -> (thread, result dict)
        self.wait_s = 0.0

    def start(self, layer, wanted):
        """wanted: list of (name, rows). Starts reading in the background."""
        if layer in self.jobs or not wanted:
            return
        out = {}

        def work():
            for name, rows in wanted:
                out[name] = self.store.read(name, rows)

        th = threading.Thread(target=work, daemon=True)
        th.start()
        self.jobs[layer] = (th, out)

    def get(self, layer, wanted):
        if layer not in self.jobs:
            self.start(layer, wanted)
        if layer not in self.jobs:
            return {}
        th, out = self.jobs.pop(layer)
        t = time.perf_counter()
        th.join()
        self.wait_s += time.perf_counter() - t
        return out

    def clear(self):
        for th, _ in self.jobs.values():
            th.join()
        self.jobs.clear()
