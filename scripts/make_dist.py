"""Pack a finished engine build into llama/dist/ for the installer. usage: python scripts/make_dist.py KEY FOLDER
KEY    windows-x86_64, windows-arm64, linux-x86_64 or linux-arm64
FOLDER a folder that holds cale_llama and the ggml/llama libraries it loads (flat, no sub-folders)
Windows archives are .zip, Linux ones .tar.gz (it keeps the links between the versions of a library). The GPU backend is not packed:
the installer downloads the official one (cale/install.py, GPU_BACKENDS).
"""
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
key, folder = sys.argv[1], Path(sys.argv[2])
skip = ("ggml-vulkan", "ggml-cuda", "ggml-rpc")
files = sorted(f for f in folder.iterdir() if (f.is_file() or f.is_symlink()) and not any(f.name.startswith(s) or f.name.startswith("lib" + s) for s in skip)
               and (f.suffix.lower() in (".dll", ".so", ".dylib") or ".so." in f.name))
if not any(f.name in ("cale_llama.dll", "libcale_llama.so") for f in files):
    sys.exit(f"{folder} has no cale_llama library")
out = ROOT / "llama" / "dist"
out.mkdir(parents=True, exist_ok=True)
if key.startswith("windows"):
    target = out / f"{key}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            z.write(f, f.name)
else:
    target = out / f"{key}.tar.gz"
    with tarfile.open(target, "w:gz", compresslevel=9) as t:
        for f in files:
            t.add(f, arcname=f.name)
print(f"{target.relative_to(ROOT)}: {len(files)} files, {target.stat().st_size / 1e6:.1f} MB")
