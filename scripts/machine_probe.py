"""Measure the basic costs that decide how fast Cale starts, apart from the AI model. Run it on two computers and compare the numbers.

usage: python scripts/machine_probe.py [--node NODE] [--out FILE.json] [--net-url URL]
Each line is one cost. The ratio between a slow and a fast computer tells where a gap comes from:
  - program start, Node start: how fast a new process starts (Windows is slower than macOS at this, and a virus scanner adds to it)
  - create / read many small files: the cost of FILES (a virus scanner and the file system). Cale's chat is tens of thousands of files.
  - one CPU thread: how fast a single thread runs. Starting Python or Node is a single thread.
  - disk write / read: how fast big files move
  - internet: download speed from a public file server (two computers on the same network should match)
Nothing here uses the AI model. It writes only to a temporary folder, and removes it.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NET_URL = "https://huggingface.co/mlx-community/gemma-3n-E4B-it-4bit/resolve/main/tokenizer.json"      # 33 MB, public


def median_run(cmd, n=5, **kw):
    times = []
    for _ in range(n):
        t = time.perf_counter()
        subprocess.run(cmd, capture_output=True, **kw)
        times.append(time.perf_counter() - t)
    return round(statistics.median(times), 3)


def files(tmp, n=3000):
    d = Path(tmp) / "many"
    d.mkdir()
    blobs = [hashlib.shake_256(str(i).encode()).digest(300 + i % 4000) for i in range(n)]
    t = time.perf_counter()
    for i, b in enumerate(blobs):
        (d / f"sub{i % 40}").mkdir(exist_ok=True) if i < 40 else None
        (d / f"sub{i % 40}" / f"f{i}.js").write_bytes(b)
    create = time.perf_counter() - t
    t = time.perf_counter()
    for i in range(n):
        (d / f"sub{i % 40}" / f"f{i}.js").read_bytes()
    read = time.perf_counter() - t
    t = time.perf_counter()
    shutil.copytree(d, Path(tmp) / "many2")
    copy = time.perf_counter() - t
    t = time.perf_counter()
    shutil.rmtree(d)
    shutil.rmtree(Path(tmp) / "many2")
    remove = time.perf_counter() - t
    return {"create_files_per_s": round(n / create), "read_files_per_s": round(n / read), "copy_files_per_s": round(n / copy), "delete_files_per_s": round(2 * n / remove)}


def disk(tmp, mb=256):
    p = Path(tmp) / "big.bin"
    chunk = os.urandom(8 << 20)
    t = time.perf_counter()
    with open(p, "wb") as f:
        for _ in range(mb // 8):
            f.write(chunk)
        f.flush()
        os.fsync(f.fileno())
    w = mb / (time.perf_counter() - t)
    t = time.perf_counter()
    with open(p, "rb") as f:
        while f.read(8 << 20):
            pass
    r = mb / (time.perf_counter() - t)             # may come from the file cache: an upper bound
    p.unlink()
    return {"disk_write_mb_s": round(w), "disk_read_cached_mb_s": round(r)}


def cpu():
    t = time.perf_counter()
    x = 0
    for i in range(3_000_000):
        x += i * i % 7
    loop = time.perf_counter() - t
    data = os.urandom(64 << 20)
    t = time.perf_counter()
    hashlib.sha256(data).hexdigest()
    sha = (64 / (time.perf_counter() - t))
    return {"python_loop_3M_s": round(loop, 3), "sha256_mb_s": round(sha)}


def net(url):
    try:
        t = time.perf_counter()
        n = 0
        try:
            import certifi
            import ssl
            ctx = ssl.create_default_context(cafile=certifi.where())       # some Python builds have no certificate list of their own
        except Exception:
            ctx = None
        with urllib.request.urlopen(url, timeout=60, context=ctx) as r:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                n += len(b)
        return {"internet_mb_s": round(n / 1e6 / (time.perf_counter() - t), 1), "internet_bytes": n}
    except Exception as e:
        return {"internet_error": f"{type(e).__name__}: {str(e)[:60]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", default=os.environ.get("CALE_NODE") or shutil.which("node"))
    ap.add_argument("--out")
    ap.add_argument("--net-url", default=NET_URL)
    a = ap.parse_args()
    out = {"python": sys.version.split()[0], "platform": sys.platform, "cpus": os.cpu_count()}
    out["python_start_s"] = median_run([sys.executable, "-c", "pass"])
    out["python_import_cale_server_s"] = median_run([sys.executable, "-c", "import cale.server"], cwd=ROOT)
    out["python_import_numpy_s"] = median_run([sys.executable, "-c", "import numpy"]) if importlib.util.find_spec("numpy") else None
    if a.node:
        out["node_start_s"] = median_run([a.node, "-e", "0"])
    tmp = tempfile.mkdtemp(prefix="cale-probe-")
    try:
        out.update(files(tmp))
        out.update(disk(tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out.update(cpu())
    out.update(net(a.net_url))
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
