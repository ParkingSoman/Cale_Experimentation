#!/usr/bin/env python3
"""Build the one-download installer for one OS.

  python scripts/build_installer.py mac            -> dist/Cale-mac.dmg  (Cale.app inside, also dist/Cale-mac.zip)
  python scripts/build_installer.py linux          -> dist/Cale-linux-x86_64.run   (--arch arm64 for ARM)
  python scripts/build_installer.py win            -> dist/win/  (payload + build_exe.ps1: run that on a Windows computer)

What goes in (a "payload"): cale/, the files of nest/ that Cale imports, llama/prepare_gguf.py, the engine library for
this OS (Windows, Linux), the demo apps, and a copy of uv (so no Python or network step is needed before Cale can start).
The AI model is NOT in the download. The installer fetches it (4.4-5.5 GB) and shows a progress bar.

The person double-clicks. A launcher starts the Cale Hub (cale/hub). The Hub installs Cale with a progress bar, then shows
what Cale and the apps are doing. Next time, the launcher starts the installed Cale straight away.
"""
import argparse
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cale.install import NEST_FILES, UV_BASE  # noqa: E402

DIST = ROOT / "dist"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "data", "node_modules", "tests", ".playwright-mcp")
TRIPLES = {("mac", "arm64"): "aarch64-apple-darwin", ("linux", "x86_64"): "x86_64-unknown-linux-gnu",
           ("linux", "arm64"): "aarch64-unknown-linux-gnu", ("win", "x86_64"): "x86_64-pc-windows-msvc",
           ("win", "arm64"): "aarch64-pc-windows-msvc"}


def get_uv(os_name, arch):
    """The uv program for the target, downloaded once and kept in dist/cache."""
    triple = TRIPLES.get((os_name, arch))
    if not triple:
        sys.exit(f"no uv for {os_name} {arch}")
    ext = "zip" if os_name == "win" else "tar.gz"
    cache = DIST / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"uv-{triple}.{ext}"
    if not archive.exists():
        print(f"downloading uv for {triple}")
        subprocess.run(["curl", "-fL", "--retry", "3", "-o", str(archive), f"{UV_BASE}uv-{triple}.{ext}"], check=True)
    return archive, ext


NODE_VERSION = "22.11.0"                       # the Meeting Recap demo app is Node: carry a copy so no one has to install it
NODE_NAMES = {("mac", "arm64"): "darwin-arm64", ("linux", "x86_64"): "linux-x64", ("linux", "arm64"): "linux-arm64",
              ("win", "x86_64"): "win-x64", ("win", "arm64"): "win-arm64"}


def get_node(os_name, arch, dest_bin):
    """Put a portable node into dest_bin (downloaded once, kept in dist/cache)."""
    name = f"node-v{NODE_VERSION}-{NODE_NAMES[(os_name, arch)]}"
    ext = "zip" if os_name == "win" else "tar.gz"
    cache = DIST / "cache"
    archive = cache / f"{name}.{ext}"
    if not archive.exists():
        print(f"downloading node {NODE_VERSION} for {NODE_NAMES[(os_name, arch)]}")
        subprocess.run(["curl", "-fL", "--retry", "3", "-o", str(archive), f"https://nodejs.org/dist/v{NODE_VERSION}/{name}.{ext}"], check=True)
    exe = "node.exe" if os_name == "win" else "node"
    wanted = f"{name}/{exe}" if os_name == "win" else f"{name}/bin/node"
    if ext == "zip":
        with zipfile.ZipFile(archive) as z:
            (dest_bin / exe).write_bytes(z.read(wanted))
    else:
        with tarfile.open(archive) as t:
            (dest_bin / exe).write_bytes(t.extractfile(wanted).read())
    (dest_bin / exe).chmod(0o755)


def stage(os_name, arch, dest):
    """Fill dest with the payload."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    shutil.copytree(ROOT / "cale", dest / "cale", ignore=IGNORE)
    (dest / "nest").mkdir()
    for f in NEST_FILES:
        shutil.copy2(ROOT / "nest" / f, dest / "nest" / f)
    (dest / "llama").mkdir()
    shutil.copy2(ROOT / "llama" / "prepare_gguf.py", dest / "llama" / "prepare_gguf.py")
    if os_name == "mac":
        (dest / "results").mkdir()
        shutil.copy2(ROOT / "results" / "g3n-curve.json", dest / "results" / "g3n-curve.json")      # a Mac's measured curve
    else:
        found = [p for p in (ROOT / "llama" / "dist").iterdir() if p.name.startswith(f"{'windows' if os_name == 'win' else 'linux'}-{arch}")]
        if not found:
            sys.exit(f"no engine archive for {os_name}-{arch} in llama/dist/")
        (dest / "llama" / "dist").mkdir()
        for p in found:
            shutil.copy2(p, dest / "llama" / "dist" / p.name)
    shutil.copytree(ROOT / "demo" / "apps", dest / "demo" / "apps", ignore=IGNORE)
    (dest / "demo" / "apps" / "chat").mkdir()                       # the Chat app is Open WebUI: only its small launcher ships (Open WebUI itself is installed separately)
    for f in ("cale_webui.py", "cale.js"):
        shutil.copy2(ROOT / "apps" / "open-webui" / f, dest / "demo" / "apps" / "chat" / f)
    (dest / "bin").mkdir()
    archive, ext = get_uv(os_name, arch)
    exe = "uv.exe" if os_name == "win" else "uv"
    if ext == "zip":
        with zipfile.ZipFile(archive) as z:
            for n in z.namelist():
                if Path(n).name == exe:
                    (dest / "bin" / exe).write_bytes(z.read(n))
    else:
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                if Path(m.name).name == exe and m.isfile():
                    (dest / "bin" / exe).write_bytes(t.extractfile(m).read())
    (dest / "bin" / exe).chmod(0o755)
    get_node(os_name, arch, dest / "bin")
    if os_name == "mac":
        shutil.copy2(compile_mac("CaleWindow"), dest / "bin" / "CaleWindow")     # shows Cale's chat and downloaded apps in their own window


def compile_mac(name):
    """Compile cale/host/mac/<name>.swift (CaleWindow, or Cale for the setup window). Returns the program's path."""
    out = DIST / "cache" / "host"
    out.mkdir(parents=True, exist_ok=True)
    src = ROOT / "cale" / "host" / "mac" / f"{name}.swift"
    exe = out / name
    if not exe.exists() or exe.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(["swiftc", "-O", "-target", "arm64-apple-macos14", "-o", str(exe), str(src)], check=True)
    return exe


# The launcher is the same idea everywhere: if Cale is installed, run its Hub with its own Python.
# If not, run the Hub from this download with uv's Python, and let it install Cale (--setup).
MAC_LAUNCHER = r'''#!/bin/sh
here="$(cd "$(dirname "$0")/../Resources/payload" && pwd)"
home="${CALE_HOME:-$HOME/Library/Application Support/Cale}"
port="${CALE_HUB_PORT:-47800}"
if curl -s -o /dev/null "http://127.0.0.1:$port/api/state"; then open "http://127.0.0.1:$port/"; exit 0; fi   # the Hub already runs
export PATH="$home/bin:$PATH"
if [ -x "$home/venv/bin/python" ] && [ -f "$home/runtime/cale/hub/server.py" ]; then
  CALE_HOME="$home" "$home/venv/bin/python" "$here/cale/install.py" --update >/dev/null 2>&1    # this download's code replaces the installed code
  CALE_HOME="$home" PYTHONPATH="$home/runtime" exec "$home/venv/bin/python" -m cale.hub
fi
cd "$here" && exec "$here/bin/uv" run --python 3.12 --no-project python -m cale.hub --setup
'''

LINUX_LAUNCHER = r'''#!/bin/sh
here="$(cd "$(dirname "$0")" && pwd)"
home="${XDG_DATA_HOME:-$HOME/.local/share}/cale"
if [ -x "$home/venv/bin/python" ] && [ -f "$home/runtime/cale/hub/server.py" ]; then
  CALE_HOME="$home" "$home/venv/bin/python" "$here/cale/install.py" --update >/dev/null 2>&1
  PATH="$home/bin:$PATH" CALE_HOME="$home" PYTHONPATH="$home/runtime" exec "$home/venv/bin/python" -m cale.hub
fi
cd "$here" && PATH="$home/bin:$PATH" exec "$here/bin/uv" run --python 3.12 --no-project python -m cale.hub --setup
'''

WIN_LAUNCHER = r'''@echo off
setlocal
set "HOMEDIR=%LOCALAPPDATA%\Cale"
if defined CALE_HOME set "HOMEDIR=%CALE_HOME%"
set "HERE=%~dp0"
curl.exe -s -o nul http://127.0.0.1:47800/api/state && ( start "" http://127.0.0.1:47800/ & exit /b )
if exist "%HOMEDIR%\venv\Scripts\python.exe" if exist "%HOMEDIR%\runtime\cale\hub\server.py" (
  set "PATH=%HOMEDIR%\bin;%PATH%"
  set "CALE_HOME=%HOMEDIR%"
  "%HOMEDIR%\venv\Scripts\python.exe" "%HERE%cale\install.py" --update >nul 2>&1
  set "PYTHONPATH=%HOMEDIR%\runtime"
  "%HOMEDIR%\venv\Scripts\python.exe" -m cale.hub
  exit /b
)
cd /d "%HERE%"
"%HERE%bin\uv.exe" run --python 3.12 --no-project python -m cale.hub --setup
'''

WIN_SETUP_CMD = r'''@echo off
title Cale
echo Starting Cale. One moment...
set "D=%LOCALAPPDATA%\Cale\app"
if not exist "%D%" mkdir "%D%"
tar -xf "%~dp0payload.zip" -C "%D%"
call "%D%\launch.cmd"
'''

WIN_SETUP_VBS = r'''Set sh = CreateObject("WScript.Shell")
here = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
sh.Run "cmd /c """"" & here & "\setup.cmd""""", 7, True
'''


INFO_PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Cale</string>
  <key>CFBundleDisplayName</key><string>Cale</string>
  <key>CFBundleIdentifier</key><string>ai.cale.app</string>
  <key>CFBundleExecutable</key><string>Cale</string>
  <key>CFBundleIconFile</key><string>Cale</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.productivity</string>
  <key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/></dict>
</dict></plist>
"""


def make_icon(path, bg=(47, 93, 80), fg=(247, 245, 238), glyph="ring"):
    """A plain icon, drawn with the standard library: a green rounded square with a white ring ("ring")
    or a kale leaf ("kale": Cale's own icon)."""
    import math
    import struct
    import zlib
    n = 1024
    rows = []
    c = n / 2
    half, r = 452, 190                                              # the square's half width, and the corner radius
    ix = half - r
    cos45 = sin45 = math.sqrt(0.5)
    L, W = 300, 175                                                 # the kale leaf: half length, half width
    for y in range(n):
        row = bytearray([0])
        for x in range(n):
            dx, dy = abs(x - c), abs(y - c)
            inside = dx <= half and dy <= half and (dx <= ix or dy <= ix or (dx - ix) ** 2 + (dy - ix) ** 2 <= r * r)
            if not inside:
                row += bytes((0, 0, 0, 0))
                continue
            if glyph == "kale":
                u = ((x - c) * cos45 + (c - y) * sin45) + 20            # along the leaf (up and to the right)
                v = -(x - c) * sin45 + (c - y) * cos45                  # across the leaf
                shape = abs(v) <= W * math.sin(math.pi * (u + L) / (2 * L)) if -L <= u <= L else False
                rib = abs(v) <= 9 and -L - 110 <= u <= L * 0.75            # the stem and the middle vein
                veins = shape and u > -L * 0.5 and abs(abs(v) - (u + L * 0.5) * 0.42) <= 7 and abs(v) < W * 0.8
                stem = abs(v) <= 16 and -L - 120 <= u < -L
                mark = (shape and not (rib or veins)) or stem
                row += bytes((*fg, 255)) if mark else bytes((*bg, 255))
                continue
            d = ((x - c) ** 2 + (y - c) ** 2) ** 0.5
            ring = 215 <= d <= 300 and not (x > c and abs(y - c) < 120)      # a "C": a ring with an opening on the right
            row += bytes((*fg, 255)) if ring else bytes((*bg, 255))
        rows.append(bytes(row))
    raw = zlib.compress(b"".join(rows), 6)

    def chunk(t, d):
        body = struct.pack(">I", len(d)) + t + d
        return body + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 6, 0, 0, 0)) + chunk(b"IDAT", raw) + chunk(b"IEND", b""))


def build_mac():
    app = DIST / "mac" / "Cale.app"
    shutil.rmtree(DIST / "mac", ignore_errors=True)
    (app / "Contents" / "MacOS").mkdir(parents=True)
    (app / "Contents" / "Resources").mkdir(parents=True)
    stage("mac", "arm64", app / "Contents" / "Resources" / "payload")
    (app / "Contents" / "Info.plist").write_text(INFO_PLIST)
    launcher = app / "Contents" / "MacOS" / "Cale"
    shutil.copy2(compile_mac("Cale"), launcher)               # a real program with its own window (cale/host/mac/Cale.swift)
    launcher.chmod(0o755)
    png = DIST / "mac" / "icon.png"
    make_icon(png, glyph="kale")
    iconset = DIST / "mac" / "Cale.iconset"
    iconset.mkdir()
    for size in (16, 32, 64, 128, 256, 512, 1024):
        out = iconset / (f"icon_{size}x{size}.png")
        subprocess.run(["sips", "-z", str(size), str(size), str(png), "--out", str(out)], check=True, capture_output=True)
        if size in (32, 64, 256, 512, 1024):
            half = size // 2
            shutil.copy2(out, iconset / f"icon_{half}x{half}@2x.png")
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(app / "Contents" / "Resources" / "Cale.icns")], check=True)
    shutil.rmtree(iconset)
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app)], check=True, capture_output=True)      # ad hoc: no developer account
    dmg = DIST / "Cale-mac.dmg"
    dmg.unlink(missing_ok=True)
    subprocess.run(["hdiutil", "create", "-volname", "Cale", "-srcfolder", str(app), "-ov", "-format", "UDZO", str(dmg)], check=True, capture_output=True)
    zip_path = DIST / "Cale-mac.zip"
    zip_path.unlink(missing_ok=True)
    subprocess.run(["ditto", "-c", "-k", "--keepParent", str(app), str(zip_path)], check=True)
    return dmg


def build_linux(arch):
    work = DIST / f"linux-{arch}"
    payload = work / "payload"
    stage("linux", arch, payload)
    launcher = payload / "launch.sh"
    launcher.write_text(LINUX_LAUNCHER)
    launcher.chmod(0o755)
    tgz = work / "payload.tar.gz"
    with tarfile.open(tgz, "w:gz") as t:
        t.add(payload, arcname=".")
    header = ('#!/bin/sh\n# Cale for Linux. Run:  sh Cale-linux.run   (or make it executable and double-click it)\n'
              'set -e\nd="${TMPDIR:-/tmp}/cale-setup-$$"\nmkdir -p "$d"\n'
              'tail -n +__SKIP__ "$0" | tar xz -C "$d"\nsh "$d/launch.sh"\nrc=$?\nrm -rf "$d"\nexit $rc\n')
    skip = header.count("\n") + 1
    header = header.replace("__SKIP__", str(skip))
    out = DIST / f"Cale-linux-{arch}.run"
    out.write_bytes(header.encode() + tgz.read_bytes())
    out.chmod(0o755)
    return out


def build_win(arch):
    """On a Mac or Linux this prepares the files. The .exe itself needs IExpress, so run work/build_exe.ps1 on Windows."""
    work = DIST / "win"
    payload = work / "payload"
    shutil.rmtree(work, ignore_errors=True)
    stage("win", arch, payload)
    (payload / "launch.cmd").write_text(WIN_LAUNCHER.replace("\n", "\r\n"))
    zip_path = work / "payload.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in payload.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(payload))
    shutil.rmtree(payload)
    (work / "setup.cmd").write_text(WIN_SETUP_CMD.replace("\n", "\r\n"))
    (work / "setup.vbs").write_text(WIN_SETUP_VBS.replace("\n", "\r\n"))
    for name in ("build_exe.ps1", "prep_windows_demo.ps1"):
        shutil.copy2(Path(__file__).with_name(name), work / name)
    return work


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("os", choices=["mac", "win", "linux"])
    ap.add_argument("--arch", default=None)
    a = ap.parse_args()
    DIST.mkdir(exist_ok=True)
    if a.os == "mac":
        out = build_mac()
    elif a.os == "linux":
        out = build_linux(a.arch or "x86_64")
    else:
        out = build_win(a.arch or "x86_64")
    print(f"built: {out}")


if __name__ == "__main__":
    main()
