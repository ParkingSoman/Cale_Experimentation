"""Test the Windows self-extractor's file writing (cale/host/win/SelfExtract.cs, class Extractor) on a Windows computer.

usage: python scripts/test_extract_win.py WORKFOLDER
  It builds ExtractTest.exe with the csc.exe that comes with Windows, makes a zip of 6,000 small files (some empty, some in deep folders, one with a
  name that tries to leave the target), extracts it with 1, 3 and 8 threads, and checks every file byte for byte.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOST = ROOT / "cale" / "host" / "win"
work = Path(sys.argv[1])
shutil.rmtree(work, ignore_errors=True)
work.mkdir(parents=True)
csc = r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
exe = work / "ExtractTest.exe"
r = subprocess.run([csc, "/nologo", "/target:exe", f"/out:{exe}", "/main:CaleHost.ExtractTestProgram", "/r:System.Windows.Forms.dll", "/r:System.Drawing.dll",
                    "/r:System.IO.Compression.dll", "/r:System.IO.Compression.FileSystem.dll", str(HOST / "SelfExtract.cs"), str(HOST / "tests" / "ExtractTest.cs")], capture_output=True, text=True)
if r.returncode:
    print(r.stdout, r.stderr)
    sys.exit("the test program did not build")

data = {}
for i in range(6000):
    name = f"pkg{i % 97}/lib/sub{i % 13}/file{i}.js"
    body = b"" if i % 50 == 0 else hashlib.shake_256(str(i).encode()).digest(200 + (i % 700))
    data[name] = body
zp = work / "test.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for n, b in data.items():
        z.writestr(n, b)
    z.writestr("../escape.txt", b"must not be written outside the target")
ok = True
for threads in (1, 3, 8):
    dest = work / f"out{threads}"
    t = time.time()
    r = subprocess.run([str(exe), str(zp), str(dest), str(threads)], capture_output=True, text=True)
    took = time.time() - t
    bad = [n for n, b in data.items() if not (dest / n).is_file() or (dest / n).read_bytes() != b]
    empty_ok = all((dest / n).stat().st_size == 0 for n, b in data.items() if b == b"" and (dest / n).exists())
    escaped = (work / "escape.txt").exists() or (dest.parent / "escape.txt").exists()
    print(f"threads {threads}: exit {r.returncode}, {took:.1f} s, wrong or missing files: {len(bad)}, empty files empty: {empty_ok}, wrote outside: {escaped}")
    ok = ok and r.returncode == 0 and not bad and empty_ok and not escaped
print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
