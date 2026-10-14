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
  CALE_MODEL_URL    Windows and Linux: where to download the .gguf (default: unsloth/gemma-3n-E4B-it-GGUF, Q4_0)
  CALE_MAX_MBPS     limit the download speed (megabytes per second), to keep a shared network usable
  CALE_LIB_SRC      the cale_llama library file to install (Windows and Linux). Default: llama/dist/<os>-<arch>/
  CALE_PROGRESS=1   also print  CALE_PROGRESS {"stage": ..., "pct": ...}  lines for an app to show
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
import time
import urllib.request
import zipfile
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
from cale import oslayer  # noqa: E402

# The files of nest/ that the installed Cale imports. cale/tests/install_check.py checks this list against the real imports.
NEST_FILES = ("__init__.py", "levels3n.py", "gemma3n_levels.json", "gemma3n.py", "elastic.py", "cancel.py", "stops.py")
GGUF_URL = "https://huggingface.co/unsloth/gemma-3n-E4B-it-GGUF/resolve/main/gemma-3n-E4B-it-Q4_0.gguf"
UV_BASE = "https://github.com/astral-sh/uv/releases/latest/download/"


def say(msg):
    print(f"==> {msg}", flush=True)


def die(msg):
    print(f"Cale install stopped: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def progress(stage, pct=None, **extra):
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
        t0, last, got = time.time(), 0.0, 0
        with open(part, "ab" if have else "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                now = time.time()
                if now - last > 0.25:
                    last = now
                    progress("model", done=have + got, total=total or 0)
                if cap and got / max(now - t0, 1e-6) > cap:
                    time.sleep(len(chunk) / cap)
        r.close()
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
    for p in (shutil.which("uv"), home / "bin" / exe):
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


def is_prepared(path):
    """True if this GGUF already has the key cale.ffn_down_split (the file is in Cale's layout)."""
    with open(path, "rb") as f:
        return b"cale.ffn_down_split" in f.read(64 * 1024 * 1024)       # the metadata is at the start of the file


def prepare(home, raw, model):
    """Lay each down matrix out as two halves, so the CPU engine can keep its fast weight repacking on (llama/prepare_gguf.py).
    If that fails, the downloaded file still works, only slower: use it as it is."""
    progress("convert", 96)
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


def write_command(home, port):
    (home / "bin").mkdir(parents=True, exist_ok=True)
    py, rt = oslayer.venv_python(home / "venv"), home / "runtime"
    if oslayer.WIN:
        (home / "bin" / "cale.cmd").write_text(
            f'@echo off\r\nset "CALE_HOME={home}"\r\nset "PYTHONPATH={rt}"\r\n"{py}" -m cale.cli %*\r\n')
        return home / "bin" / "cale.cmd"
    exe = home / "bin" / "cale"
    exe.write_text(f'#!/bin/sh\nexport CALE_HOME="{home}" PYTHONPATH="{rt}"\nexec "{py}" -m cale.cli "$@"\n')
    exe.chmod(0o755)
    link = Path.home() / ".local" / "bin"
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


def main():
    backend = choose_backend()
    home = oslayer.home_dir()
    port = int(os.environ.get("CALE_PORT", "47821"))
    os.environ["CALE_HOME"] = str(home)
    for sub in ("runtime", "models", "logs", "bin"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    say(f"Installing Cale for {oslayer.NAME} ({backend} engine) in {home}")

    progress("runtime", 2)
    say("Copying the runtime")
    copy_runtime(home)

    progress("python", 8)
    say("Setting up Python (its own copy, nothing shared with other projects)")
    uv = find_uv(home)
    if not oslayer.venv_python(home / "venv").exists():
        subprocess.run([uv, "venv", "--python", "3.12", str(home / "venv")], check=True, stdout=subprocess.DEVNULL)
    req = SRC / "cale" / f"requirements-{backend}.txt"
    subprocess.run([uv, "pip", "install", "--quiet", "--python", str(oslayer.venv_python(home / "venv")), "-r", str(req)], check=True)

    progress("model", 25)
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
    if oslayer.MAC:
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
    progress("done", 100)
    print(f"Cale is running at http://127.0.0.1:{port}")
    print(f"Check it any time with:  {cmd} status")


if __name__ == "__main__":
    main()
