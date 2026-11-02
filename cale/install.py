#!/usr/bin/env python3
"""Install Cale on this computer: Mac, Windows or Linux. Run once:  python cale/install.py

Standard library only, so any Python 3.9 or newer can run it. Everything goes in one folder.
Nothing outside it changes, except on a Mac: a LaunchAgent (Cale starts at login) and a `cale` link.
On Windows and Linux there is no service: apps start Cale when they need it (cale.client.ensure_running).

Settings (all optional, as environment variables):
  CALE_HOME         install folder (default: the usual place for this OS, see cale/oslayer.py)
  CALE_MODEL_SRC    a local model (a folder on Mac, a .gguf file on Windows and Linux) to link in, no extra disk
  CALE_ALLOW_COPY=1 copy the model if a hard link is not possible
  CALE_MODEL_REPO   Mac: the Hugging Face repo (default mlx-community/gemma-3n-E4B-it-4bit)
  CALE_MODEL_URL    Windows and Linux: where to download the .gguf (default: parkingsoman/cale-gemma-3n-E4B-it-GGUF, already in Cale layout)
  CALE_MAX_MBPS     limit the download speed (megabytes per second), to keep a shared network usable
  CALE_LIB_SRC      the cale_llama library file to install (Windows and Linux). Default: llama/dist/<os>-<arch>/
  CALE_PROGRESS=1   also print  CALE_PROGRESS {"stage": ..., "pct": ...}  lines for an app to show
  CALE_NO_PATH=1    Windows, for tests: do not add Cale's bin folder to this user's PATH (and CALE_NO_LAUNCH_AGENT=1 on a Mac)
  CALE_DEBUG=1      test mode (Mac): CALE_AUTO_APPROVE=1 skips the Allow dialog
"""
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
from cale import oslayer  # noqa: E402

# The files of nest/ that the installed Cale imports. cale/tests/install_check.py checks this list against the real imports.
NEST_FILES = ("__init__.py", "levels3n.py", "gemma3n_levels.json", "gemma3n.py", "elastic.py", "cancel.py", "stops.py")
GGUF_URL = "https://huggingface.co/parkingsoman/cale-gemma-3n-E4B-it-GGUF/resolve/main/gemma-3n-E4B-it-Q4_0.cale.gguf"
UV_BASE = "https://github.com/astral-sh/uv/releases/latest/download/"


def say(msg):
    print(f"==> {msg}", flush=True)


def die(msg):
    print(f"Cale install stopped: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def progress(stage, pct=None, **extra):
    try:
        from cale import timeline
        timeline.mark("install_" + stage)        # when each stage starts: the length of a stage is the gap to the next (scripts/timeline_report.py --anchor install_runtime)
        if stage == "done":
            timeline.flush()
    except Exception:
        pass
    if os.environ.get("CALE_PROGRESS"):
        print("CALE_PROGRESS " + json.dumps({"stage": stage, **({"pct": pct} if pct is not None else {}), **extra}), flush=True)


def arch():
    m = platform.machine().lower()
    return "arm64" if m in ("arm64", "aarch64") else "x86_64" if m in ("x86_64", "amd64") else m


def choose_backend():
    a = arch()
    if oslayer.MAC:
        if a != "arm64":
            die("Cale needs a Mac with Apple silicon.")
        if int(platform.mac_ver()[0].split(".")[0] or 0) < 14:
            die("Cale needs macOS 14 or newer.")
        return "mlx"
    if a not in ("x86_64", "arm64"):
        die(f"Cale does not support {oslayer.NAME} on {a} yet. It needs a 64-bit Intel, AMD or ARM computer.")
    return "llama"


def download(url, dest, expect_sha=None):
    """Download with resume. Prints progress. Keeps `dest`.part until the file is whole."""
    part = Path(str(dest) + ".part")
    have = part.stat().st_size if part.exists() else 0
    cap = float(os.environ.get("CALE_MAX_MBPS") or 0) * 1e6
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        r = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:                       # the part file is already complete
            r = None
        else:
            raise
    total = None
    if r is not None:
        if r.status == 206:
            total = int(r.headers["Content-Range"].split("/")[1])
        else:
            have = 0                                     # the server sent everything again
            total = int(r.headers.get("Content-Length") or 0) or None
        expect_sha = expect_sha or r.headers.get("X-Linked-Etag", "").strip('"') or None
        t0, last, got, prev, worst_gap = time.time(), 0.0, 0, time.time(), 0.0
        with open(part, "ab" if have else "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                now = time.time()
                worst_gap, prev = max(worst_gap, now - prev), now
                if now - last > 0.25:
                    last = now
                    progress("model", done=have + got, total=total or 0)
                if cap and got / max(now - t0, 1e-6) > cap:
                    time.sleep(len(chunk) / cap)
        r.close()
        try:                                              # a passive sample: how fast this computer's connection and disk were (numbers only, see cale/samples.py)
            from cale.samples import Samples
            s_ = Samples(oslayer.home_dir() / "samples.jsonl")
            secs = max(time.time() - t0, 1e-6)
            if got >= 50_000_000:                         # a small file says nothing about speed
                s_.record("download", mb=round(got / 1e6), seconds=round(secs, 2), mb_per_s=round(got / 1e6 / secs, 2), worst_gap_s=round(worst_gap, 2),
                          resumed=bool(have), capped=bool(cap))
                s_.flush()
        except Exception:
            pass
    size = part.stat().st_size
    if total and size != total:
        die(f"the model download is incomplete ({size} of {total} bytes). Run the install again to resume.")
    if expect_sha and len(expect_sha) == 64:
        h = hashlib.sha256()
        with open(part, "rb") as f:
            for blk in iter(lambda: f.read(1 << 22), b""):
                h.update(blk)
        if h.hexdigest() != expect_sha:
            part.unlink()
            die("the downloaded model is damaged (checksum differs). Run the install again.")
    os.replace(part, dest)
    progress("model", done=size, total=size)


def find_uv(home):
    exe = "uv.exe" if oslayer.WIN else "uv"
    for p in (SRC / "bin" / exe, shutil.which("uv"), home / "bin" / exe):          # SRC/bin: the one-download installers carry their own uv
        if p and Path(p).exists():
            return str(p)
    triple = {("darwin", "arm64"): "aarch64-apple-darwin", ("linux", "x86_64"): "x86_64-unknown-linux-gnu",
              ("linux", "arm64"): "aarch64-unknown-linux-gnu", ("win32", "x86_64"): "x86_64-pc-windows-msvc",
              ("win32", "arm64"): "aarch64-pc-windows-msvc"}.get((sys.platform if sys.platform != "linux" else "linux", arch()))
    if not triple:
        die("uv is missing and has no download for this computer. Install uv from https://docs.astral.sh/uv/ and run again.")
    say("Getting uv (a small tool that sets up Python)")
    ext = "zip" if oslayer.WIN else "tar.gz"
    archive = home / f"uv.{ext}"
    urllib.request.urlretrieve(f"{UV_BASE}uv-{triple}.{ext}", archive)
    (home / "bin").mkdir(parents=True, exist_ok=True)
    if ext == "zip":
        with zipfile.ZipFile(archive) as z:
            for n in z.namelist():
                if Path(n).name in ("uv.exe", "uvx.exe"):
                    (home / "bin" / Path(n).name).write_bytes(z.read(n))
    else:
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                if Path(m.name).name in ("uv", "uvx") and m.isfile():
                    target = home / "bin" / Path(m.name).name
                    target.write_bytes(t.extractfile(m).read())
                    target.chmod(target.stat().st_mode | stat.S_IEXEC)
    archive.unlink()
    return str(home / "bin" / exe)


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        if os.environ.get("CALE_ALLOW_COPY") != "1":
            die(f"cannot link {src.name} (a different disk?). Set CALE_ALLOW_COPY=1 to copy it (about 6 GB).")
        shutil.copy2(src, dst)


def copy_runtime(home):
    rt = home / "runtime"
    for sub in ("cale", "nest"):
        shutil.rmtree(rt / sub, ignore_errors=True)
    ignore = shutil.ignore_patterns("__pycache__", "tests", "*.sh", "*.ps1", "*.plist", "requirements*.txt", "README.md")
    shutil.copytree(SRC / "cale", rt / "cale", ignore=ignore)
    (rt / "llama").mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC / "llama" / "prepare_gguf.py", rt / "llama" / "prepare_gguf.py")      # makes the model fast on a CPU
    (rt / "nest").mkdir(parents=True, exist_ok=True)
    for f in NEST_FILES:
        shutil.copy2(SRC / "nest" / f, rt / "nest" / f)
    # The one-download installers carry a Node (for the Node demo app) and the window program that shows Cale's chat and apps.
    for tool in ("node", "node.exe", "CaleWindow", "CaleWindow.exe", "WebView2Loader.dll"):
        if (SRC / "bin" / tool).exists():
            (home / "bin").mkdir(parents=True, exist_ok=True)
            shutil.copy2(SRC / "bin" / tool, home / "bin" / tool)
    apps = SRC / "demo" / "apps"                                     # the demo apps that come with the one-download installers
    if apps.is_dir():
        shutil.rmtree(home / "apps", ignore_errors=True)
        shutil.copytree(apps, home / "apps", ignore=shutil.ignore_patterns("__pycache__", "data", "node_modules", ".DS_Store"))


def chat_bundle_source():
    """Where the chat window program is (built by scripts/build_chat.py): CALE_CHAT_SRC, else chat/ or dist/chat/ next to cale/. None if absent."""
    for c in (os.environ.get("CALE_CHAT_SRC"), SRC / "chat", SRC / "dist" / "chat"):
        if c and (Path(c) / "server" / "index.js").exists():
            return Path(c)
    return None


def _long(p):
    """On Windows, a path form with no 259-character limit (the chat's packages have long names inside long folders)."""
    p = Path(p)
    if os.name != "nt":
        return p
    full = str(p.resolve())
    return Path(full if full.startswith("\\\\?\\") else "\\\\?\\" + full)


def _moves(src, home):
    """True when the chat folder is the installer's own unpacked copy (<home>/app/chat) and sits on the same disk as <home>: then it is moved, not copied.
    A developer's dist/chat (or CALE_CHAT_SRC) is never moved."""
    try:
        return (Path(src).resolve() == (home / "app" / "chat").resolve() and SRC.resolve() == (home / "app").resolve()
                and os.stat(src).st_dev == os.stat(home).st_dev)
    except OSError:
        return False


def install_chat(home):
    """Copy Cale's chat window program into <home>/chat. What the person made (<home>/chat-data) is never touched."""
    src = chat_bundle_source()
    if src is None:
        say("(No chat window program in this download: skipping it.)")
        return False
    dst = home / "chat"
    tmp = home / "chat.new"
    shutil.rmtree(_long(tmp), ignore_errors=True)
    if _moves(src, home):
        # The one-download installer unpacked the chat into <home>/app/chat. Copying 60,000 small files a second time took about 150 s on a laptop with a
        # virus scanner. A folder moved inside one disk takes no time. The unpacked copy is not needed again (a new download unpacks it again).
        try:
            os.replace(_long(src), _long(tmp))
        except OSError:
            shutil.rmtree(_long(tmp), ignore_errors=True)
            shutil.copytree(_long(src), _long(tmp), symlinks=True)
    else:
        shutil.copytree(_long(src), _long(tmp), symlinks=True)
    for link in (tmp / "server" / "storage", tmp / "collector" / "hotdir"):      # made at run time by cale/chat.py, never shipped
        if link.is_symlink() or link.exists():
            shutil.rmtree(link) if link.is_dir() and not link.is_symlink() else link.unlink()
    shutil.rmtree(_long(dst), ignore_errors=True)
    if dst.exists():                                       # Windows: a read-only file stops rmtree. Make everything writable and try again.
        for d, dirs, files in os.walk(_long(dst)):
            for n in dirs + files:
                try:
                    os.chmod(os.path.join(d, n), 0o700)
                except OSError:
                    pass
        shutil.rmtree(_long(dst))
    tmp.rename(dst)
    return True


def install_model(home, backend):
    if backend == "mlx":
        model = home / "models" / "gemma-3n-e4b"
        if (model / ".complete").exists():
            say("Model is already installed")
        elif os.environ.get("CALE_MODEL_SRC"):
            srcdir = Path(os.environ["CALE_MODEL_SRC"])
            if not (srcdir / "config.json").exists():
                die(f"model folder not found: {srcdir}")
            say("Linking the model (no extra disk space)")
            for f in srcdir.iterdir():
                if f.is_file():
                    link_or_copy(f, model / f.name)
            (model / ".complete").write_text("ok\n")
        else:
            say("Downloading the model (it resumes if interrupted)")
            py = str(oslayer.venv_python(home / "venv"))
            r = subprocess.run([py, str(home / "runtime" / "cale" / "fetch_model.py"),
                                os.environ.get("CALE_MODEL_REPO", "mlx-community/gemma-3n-E4B-it-4bit"), str(model)])
            if r.returncode:
                die("the model download failed. Run the install again to resume.")
        return model
    model = home / "models" / "gemma-3n-e4b.gguf"
    done = Path(str(model) + ".complete")
    raw = home / "models" / "gemma-3n-e4b.download.gguf"
    if done.exists() and model.exists():
        say("Model is already installed")
        return model
    if os.environ.get("CALE_MODEL_SRC"):
        s = Path(os.environ["CALE_MODEL_SRC"])
        if s.is_dir():
            s = sorted(s.glob("*.gguf"))[0] if list(s.glob("*.gguf")) else s
        if not s.is_file():
            die(f"model file not found: {s}")
        say("Using the model file you gave (no download)")
        if is_prepared(s):
            link_or_copy(s, model)                      # already in Cale's layout: nothing to convert
            done.write_text("ok\n")
            return model
        link_or_copy(s, raw)
    elif raw.exists() and not (home / "models" / "gemma-3n-e4b.download.complete").exists():
        say("Finishing the model download (it resumes)")
        download(os.environ.get("CALE_MODEL_URL", GGUF_URL), raw)
    elif not raw.exists():
        say("Downloading the model (about 4.4 GB, it resumes if interrupted)")
        raw.parent.mkdir(parents=True, exist_ok=True)
        download(os.environ.get("CALE_MODEL_URL", GGUF_URL), raw)
    (home / "models" / "gemma-3n-e4b.download.complete").write_text("ok\n")
    prepare(home, raw, model)
    done.write_text("ok\n")
    return model


def prefetch_model(home, backend):
    """Windows and Linux: start the model download now, in the background. It needs only the network, so it runs while the runtime, the chat and
    Python are set up (those need the disk and the processor). Returns (thread, errors) to wait for, or None when there is nothing to download.
    (The Mac download runs through Python's own packages, so it starts after them.)"""
    if backend != "llama" or os.environ.get("CALE_MODEL_SRC"):
        return None
    model = home / "models" / "gemma-3n-e4b.gguf"
    raw = home / "models" / "gemma-3n-e4b.download.gguf"
    complete = home / "models" / "gemma-3n-e4b.download.complete"
    if (Path(str(model) + ".complete").exists() and model.exists()) or complete.exists():
        return None
    errors = []

    def work():
        try:
            say("Downloading the model (about 4.4 GB, it resumes if interrupted)")
            raw.parent.mkdir(parents=True, exist_ok=True)
            download(os.environ.get("CALE_MODEL_URL", GGUF_URL), raw)
            complete.write_text("ok\n")
        except BaseException as e:                       # die() ends the thread with SystemExit: hand it to the main thread
            errors.append(e)

    t = threading.Thread(target=work, daemon=True)
    t.start()
    return t, errors


def wait_for(pre):
    """Wait for prefetch_model. A failed download stops the install, as it did when it ran in line."""
    if pre is None:
        return
    pre[0].join()
    if pre[1]:
        e = pre[1][0]
        if isinstance(e, SystemExit):
            raise e
        die(f"the model download failed ({type(e).__name__}). Run the install again to resume.")


def is_prepared(path):
    """True if this GGUF already has the key cale.ffn_down_split (the file is in Cale's layout)."""
    with open(path, "rb") as f:
        return b"cale.ffn_down_split" in f.read(64 * 1024 * 1024)       # the metadata is at the start of the file


def warm_chat(home):
    """Start the chat once and stop it, so Node's compile cache is full before the first open. It needs the network and no AI, so it runs while the model
    downloads. A failure costs only the speed of the first open."""
    try:
        subprocess.run([str(oslayer.venv_python(home / "venv")), "-m", "cale.chat", "--warm"], env=dict(os.environ, CALE_HOME=str(home), PYTHONPATH=str(home / "runtime")),
                       timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        say(f"(The chat start-up cache was not made: {type(e).__name__}. The first open is a little slower.)")


def prepare(home, raw, model):
    """Lay each down matrix out as two halves, so the CPU engine can keep its fast weight repacking on (llama/prepare_gguf.py).
    If that fails, the downloaded file still works, only slower: use it as it is."""
    progress("convert", 96)
    if is_prepared(raw):                                  # the hosted file is already in Cale's layout: nothing to convert (saves about a minute)
        model.unlink(missing_ok=True)
        os.replace(raw, model)
        return
    say("Preparing the model for this computer (about a minute)")
    r = subprocess.run([str(oslayer.venv_python(home / "venv")), str(home / "runtime" / "llama" / "prepare_gguf.py"),
                        str(raw), str(model)])
    if r.returncode != 0 or not model.exists():
        print("(Could not prepare the model. Cale will use it as it is, a little slower.)", flush=True)
        model.unlink(missing_ok=True)
        os.replace(raw, model)
        return
    raw.unlink()                                          # a copy of the same size: free the disk


# The official llama.cpp GPU backend for the release Cale is built from. Cale's own libraries load it at run time when a
# GPU is there (cale/backends/devices.py decides whether to use it). The file is checked against the published hash.
LLAMA_TAG = "b11558"
GPU_BACKENDS = {
    ("windows", "x86_64"): ("llama-b11558-bin-win-vulkan-x64.zip", "281886df1e632438266ef35d38ecdfdc21cdced0e1f7331035b87706dc6507f6", "ggml-vulkan.dll"),
    ("windows", "arm64"): ("llama-b11558-bin-win-vulkan-arm64.zip", "ab9c9d0373ef94cbb334fa227c1b780c7e7e1bf77c4c5b227555fd1b208c4732", "ggml-vulkan.dll"),
    ("linux", "x86_64"): ("llama-b11558-bin-ubuntu-vulkan-x64.tar.gz", "f9e975512633a6183f58ae5e9afb1870af7c5b1100c6335bc21f88e90bc1c9a5", "libggml-vulkan.so"),
    ("linux", "arm64"): ("llama-b11558-bin-ubuntu-vulkan-arm64.tar.gz", "dfccb2abdbdf4004227f576d62a7ff08debda00ad6bb4dbb97df7b47d770a59a", "libggml-vulkan.so"),
}

# NVIDIA: the official CUDA 12.4 backend (it supports cards from the Pascal generation on, which CUDA 13 does not) and its runtime.
# Measured on a GeForce MX150: prompts 68 tokens/s with CUDA, 26 with Vulkan, 16 on the CPU. Answers the same speed.
CUDA_NEEDS = 12040                                          # the CUDA version the driver must support (12.4)
CUDA_BACKEND = {
    ("windows", "x86_64"): (
        ("llama-b11558-bin-win-cuda-12.4-x64.zip", "e2fa361bbe42e1a49f5462e2b500c25a6c8b13c29a4005721cf1d98dfe28c845", ("ggml-cuda.dll",)),
        ("cudart-llama-bin-win-cuda-12.4-x64.zip", "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
         ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"))),
}


def install_library(home):
    """Put Cale's engine libraries (cale_llama and the ggml backends it loads) in home/lib. Then add the GPU backend.
    Source, first found: CALE_LIB_SRC (a library file: its folder is used), llama/dist/<os>-<arch>/ (a folder, for builders),
    llama/dist/<os>-<arch>.zip or .tar.gz (what ships)."""
    name = "cale_llama.dll" if oslayer.WIN else "libcale_llama.so"
    key = f"{'windows' if oslayer.WIN else 'linux'}-{arch()}"
    lib = home / "lib"
    src_file = os.environ.get("CALE_LIB_SRC")
    dist = SRC / "llama" / "dist"
    folder = Path(src_file).parent if src_file and Path(src_file).is_file() else dist / key
    if (folder / name).is_file():
        lib.mkdir(parents=True, exist_ok=True)
        for f in folder.iterdir():                          # the library and, in a run-time-backend build, its neighbours
            if f.is_file() and (f.suffix.lower() in (".dll", ".so", ".dylib") or ".so." in f.name):
                shutil.copy2(f, lib / f.name)
    elif (dist / f"{key}.zip").is_file():
        lib.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dist / f"{key}.zip") as z:
            for n in z.namelist():                           # flat files only: nothing may land outside the folder
                if "/" not in n and "\\" not in n and not n.startswith("."):
                    (lib / n).write_bytes(z.read(n))
    elif (dist / f"{key}.tar.gz").is_file():
        lib.mkdir(parents=True, exist_ok=True)
        with tarfile.open(dist / f"{key}.tar.gz") as t:
            for m in t.getmembers():
                if "/" not in m.name and not m.name.startswith(".") and (m.isfile() or m.issym()):
                    target = lib / m.name
                    target.unlink(missing_ok=True)
                    if m.issym():
                        if "/" not in m.linkname:
                            target.symlink_to(m.linkname)
                    else:
                        target.write_bytes(t.extractfile(m).read())
                        target.chmod(0o755)
    else:
        die(f"the {name} library for {key} was not found. Build it with llama/build.{'ps1' if oslayer.WIN else 'sh'} "
            "(CALE_BACKENDS_DL=1 for the run-time-backend build), pack it with scripts/make_dist.py, or set CALE_LIB_SRC.")
    install_gpu_backend(home)
    install_cuda_backend(home)


def install_cuda_backend(home):
    """On a computer with an NVIDIA driver that supports CUDA 12.4: add the CUDA backend (about 0.6 GB). Optional: without it Cale uses Vulkan."""
    entry = CUDA_BACKEND.get(("windows" if oslayer.WIN else "linux", arch()))
    if entry is None or os.environ.get("CALE_NO_CUDA") == "1" or not (home / "lib" / "ggml-base.dll").exists():
        return
    have = oslayer.cuda_driver_version()
    if have == 0:
        return                                               # no NVIDIA driver: nothing to add
    if have < CUDA_NEEDS:
        print(f"(An NVIDIA GPU is here, but its driver supports CUDA {have // 1000}.{have % 1000 // 10}, and Cale's CUDA backend needs 12.4. "
              "Update the NVIDIA driver to make prompts faster. Cale uses Vulkan meanwhile.)", flush=True)
        return
    if (home / "lib" / "ggml-cuda.dll").exists():
        return
    say("An NVIDIA GPU is here. Getting the CUDA backend (optional, about 0.6 GB): prompts run about 2.5 times faster than with Vulkan")
    done = []
    try:
        for asset, sha, members in entry:
            tmp = home / "models" / asset
            try:
                download(f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/{asset}", tmp, expect_sha=sha)
                with zipfile.ZipFile(tmp) as z:
                    for m in members:
                        (home / "lib" / m).write_bytes(z.read(next(n for n in z.namelist() if Path(n).name == m)))
                        done.append(m)
            finally:
                tmp.unlink(missing_ok=True)
                Path(str(tmp) + ".part").unlink(missing_ok=True)
    except (Exception, SystemExit) as e:                      # optional: Cale runs without it
        print(f"(No CUDA backend: {e}. Cale will use Vulkan.)", flush=True)
        for m in done:
            (home / "lib" / m).unlink(missing_ok=True)


def install_gpu_backend(home):
    """Add the official Vulkan backend next to the library. A GPU is optional: if this step fails, Cale runs on the CPU."""
    if os.environ.get("CALE_NO_GPU_BACKEND") == "1" or not (home / "lib" / ("ggml-base.dll" if oslayer.WIN else "libggml-base.so")).exists():
        return                                               # a build with everything inside one library has no backends to add
    entry = GPU_BACKENDS.get(("windows" if oslayer.WIN else "linux", arch()))
    target = home / "lib" / (entry[2] if entry else "")
    if not entry or target.exists():
        return
    asset, sha, member = entry
    say("Getting the GPU backend (optional, about 30 MB)")
    tmp = home / "models" / asset
    try:
        download(f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/{asset}", tmp, expect_sha=sha)
        if asset.endswith(".zip"):
            with zipfile.ZipFile(tmp) as z:
                target.write_bytes(z.read(next(n for n in z.namelist() if Path(n).name == member)))
        else:
            with tarfile.open(tmp) as t:
                m = next(m for m in t.getmembers() if Path(m.name).name == member)
                target.write_bytes(t.extractfile(m).read())
    except (Exception, SystemExit) as e:                      # download() exits on a bad download: here that is not fatal
        print(f"(No GPU backend: {e}. Cale will use the CPU.)", flush=True)
        target.unlink(missing_ok=True)
    finally:
        tmp.unlink(missing_ok=True)
        Path(str(tmp) + ".part").unlink(missing_ok=True)


def add_to_windows_path(folder):
    """Put the folder on this user's PATH, so any app can run `cale`. No admin rights: it is the user's own setting."""
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ | winreg.KEY_WRITE) as k:
        try:
            cur, kind = winreg.QueryValueEx(k, "Path")
        except FileNotFoundError:
            cur, kind = "", winreg.REG_EXPAND_SZ
        if str(folder).lower() not in [x.strip().lower() for x in cur.split(";")]:
            winreg.SetValueEx(k, "Path", 0, kind, (cur.rstrip(";") + ";" if cur else "") + str(folder))
    try:                                                         # tell open windows, so new programs see it
        import ctypes
        ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x1A, 0, "Environment", 2, 3000, None)
    except Exception:
        pass


def write_command(home, port):
    (home / "bin").mkdir(parents=True, exist_ok=True)
    py, rt = oslayer.venv_python(home / "venv"), home / "runtime"
    if oslayer.WIN:
        (home / "bin" / "cale.cmd").write_text(
            f'@echo off\r\nset "CALE_HOME={home}"\r\nset "PYTHONPATH={rt}"\r\n"{py}" -m cale.cli %*\r\n')
        if os.environ.get("CALE_NO_PATH") != "1":      # the switch is for tests: it leaves this user's real PATH alone
            add_to_windows_path(home / "bin")
        return home / "bin" / "cale.cmd"
    exe = home / "bin" / "cale"
    exe.write_text(f'#!/bin/sh\nexport CALE_HOME="{home}" PYTHONPATH="{rt}"\nexec "{py}" -m cale.cli "$@"\n')
    exe.chmod(0o755)
    link = Path.home() / ".local" / "bin"
    if not oslayer.MAC:
        link.mkdir(parents=True, exist_ok=True)                  # most Linux desktops put this folder on PATH at the next login
    if link.is_dir():
        (link / "cale").unlink(missing_ok=True)
        (link / "cale").symlink_to(exe)
    return exe


def install_launch_agent(home, port):
    plist = Path.home() / "Library" / "LaunchAgents" / "ai.cale.runtime.plist"
    extra = ""
    if os.environ.get("CALE_DEBUG") == "1":                     # test mode only: it turns off the Allow dialog on request
        extra = ("    <key>CALE_DEBUG</key><string>1</string>\n"
                 f"    <key>CALE_AUTO_APPROVE</key><string>{os.environ.get('CALE_AUTO_APPROVE', '0')}</string>")
    if port != 47821:
        extra += f"\n    <key>CALE_PORT</key><string>{port}</string>"
    from xml.sax.saxutils import escape
    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_text((SRC / "cale" / "ai.cale.runtime.plist").read_text().replace("__HOME__", escape(str(home))).replace("__EXTRA_ENV__", extra))
    return plist


def update():
    """Bring an installed Cale up to this download's code: runtime, demo apps, tools. No model, no tuning, a few seconds.
    The running Cale keeps its old code until it restarts (cale restart)."""
    home = oslayer.home_dir()
    if not (home / "installed.ok").exists():
        return 1
    os.environ["CALE_HOME"] = str(home)
    copy_runtime(home)
    try:
        if choose_backend() == "llama":
            install_library(home)
    except (OSError, SystemExit):                            # a library that a running Cale holds open cannot be replaced: keep the old one
        pass
    return 0


def main():
    if "--update" in sys.argv:
        sys.exit(update())
    backend = choose_backend()
    home = oslayer.home_dir()
    port = int(os.environ.get("CALE_PORT", "47821"))
    os.environ["CALE_HOME"] = str(home)
    for sub in ("runtime", "models", "logs", "bin"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    say(f"Installing Cale for {oslayer.NAME} ({backend} engine) in {home}")

    pre = prefetch_model(home, backend)           # Windows and Linux: the download starts now and runs while the next steps set Cale up

    progress("runtime", 2)
    say("Copying the runtime")
    copy_runtime(home)

    progress("chat", 6)
    say("Setting up the chat window")
    install_chat(home)

    progress("python", 8)
    say("Setting up Python (its own copy, nothing shared with other projects)")
    uv = find_uv(home)
    if not oslayer.venv_python(home / "venv").exists():
        subprocess.run([uv, "venv", "--python", "3.12", str(home / "venv")], check=True, stdout=subprocess.DEVNULL)
    req = SRC / "cale" / f"requirements-{backend}.txt"
    subprocess.run([uv, "pip", "install", "--quiet", "--python", str(oslayer.venv_python(home / "venv")), "-r", str(req)], check=True)

    warm_chat(home)                                       # Node keeps its compiled code of the chat: the first open skips the compile (about 2x faster chat start)
    progress("model", 25)
    wait_for(pre)
    install_model(home, backend)
    if backend == "llama":
        install_library(home)
    elif (SRC / "results" / "g3n-curve.json").exists():
        shutil.copy2(SRC / "results" / "g3n-curve.json", home / "curve.json")      # numbers measured on a Mac: only copied to a Mac

    cmd = write_command(home, port)
    progress("tune", 80)
    say("Getting Cale ready for this computer. This takes a few minutes and happens once.")
    tuned = subprocess.run([str(oslayer.venv_python(home / "venv")), "-m", "cale.tune"], env=dict(os.environ, CALE_HOME=str(home), PYTHONPATH=str(home / "runtime")),
                           stdout=open(home / "logs" / "tune.log", "w"), stderr=subprocess.STDOUT)
    if tuned.returncode != 0:
        say("(Cale will finish getting ready the first time an app asks.)")
    else:
        try:
            problems = json.loads((home / "tune-report.json").read_text()).get("problems") or []
        except Exception:
            problems = []
        if problems:
            say("(Some measurements did not finish. Cale will repeat them the first time an app asks. Details: logs/tune.log)")
    progress("service", 95)
    if oslayer.MAC and os.environ.get("CALE_NO_LAUNCH_AGENT") != "1":      # the switch is for tests: it leaves the real LaunchAgent alone
        say("Starting Cale at login")
        install_launch_agent(home, port)
        from cale import service
        service.stop()
        ok, msg = service.start(60)
    else:
        say("Checking that Cale starts (apps start it themselves when they need it)")
        from cale import service
        ok, msg = service.start(60)
    if not ok:
        die(msg)
    (home / "installed.ok").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))      # the Cale Hub counts Cale as installed only with this file
    progress("done", 100)
    print(f"Cale is running at http://127.0.0.1:{port}")
    print(f"Check it any time with:  {cmd} status")


if __name__ == "__main__":
    main()
