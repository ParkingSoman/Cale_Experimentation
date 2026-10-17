#!/usr/bin/env python3
"""Build Cale's chat window bundle: AnythingLLM (MIT licence) rebranded as Cale, ready to copy into an install.

    python scripts/build_chat.py --src <AnythingLLM source folder> --node-dir <folder with bin/node> [--out dist/chat]

--src       a checkout of AnythingLLM (tested: v1.18.0): the folder that holds server/, collector/, frontend/
--node-dir  an unpacked Node (the one Cale ships, 22.x): <node-dir>/bin/node. It must be the same Node that runs the bundle.
--yarn      a yarn 1 command (default: yarn on the PATH, else installed with npm into the work folder)
--work      a scratch folder (default: <out>.work). Deleted at the end unless --keep-work.
--out       the bundle (default dist/chat)

The build needs the internet once (npm packages, the small embedding file). The bundle then works offline.

Result (what cale/chat.py expects):
    <out>/server/            AnythingLLM server (production packages, the built and rebranded frontend in server/public)
    <out>/collector/         the document reader (text, PDF, Word, ...; no web page reader: no Chromium download)
    <out>/template.db        an empty database with all tables made (no Prisma command is needed on the person's computer)
    <out>/models-seed/       the small file that turns text into numbers (about 23 MB), so documents work offline
    <out>/preload.js         keeps every server on this computer
    <out>/LICENSE-AnythingLLM.txt
    <out>/bundle.json        what is in it

Install size and the exact installer change: see the end of the run (it prints the size).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

EMBEDDER = "Xenova/all-MiniLM-L6-v2"
SKIP = shutil.ignore_patterns("node_modules", "dist", ".git", "storage", "hotdir", "__tests__", ".env*", "*.log")


def top_only(*names):
    """An ignore rule for copytree that skips these names in the top folder only (node_modules has folders with the same names)."""
    def ignore(folder, entries):
        return [e for e in entries if e in names] if Path(folder).name in ("server", "collector") and Path(folder).parent.name != "node_modules" else []
    return ignore


# Packages the chat window never runs (cloud providers, other vector databases, the tool that makes the database, PDF export).
# The build test (cale/tests or the end-to-end run in the report) must pass after this. Add to the list only with that test.
UNUSED = ("prisma", "@prisma/engines", "@mintplex-labs/mdpdf-*", "chromadb", "chromadb-default-embed", "@azure")


def prune(bundle):
    """Remove what is not needed on this computer: unused packages, other systems' native files, maps, notes, type files."""
    import glob
    import platform
    osname = {"darwin": "darwin", "win32": "win32"}.get(sys.platform, "linux")
    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
    for sub in ("server", "collector"):
        nm = bundle / sub / "node_modules"
        if sub == "server":
            for pat in UNUSED:
                for hit in glob.glob(str(nm / pat)):
                    shutil.rmtree(hit)
        ort = nm / "onnxruntime-node" / "bin" / "napi-v3"                       # the number engine: one folder per system and chip
        if ort.is_dir():
            for d in ort.iterdir():
                if d.name != osname:
                    shutil.rmtree(d)
                else:
                    for chip in d.iterdir():
                        if chip.name != arch:
                            shutil.rmtree(chip)
        for dirpath, dirs, files in os.walk(nm):
            for f in files:
                if f.endswith((".map", ".d.ts", ".md")) and not f.upper().startswith("LICENSE"):
                    try:
                        os.remove(os.path.join(dirpath, f))
                    except OSError:
                        pass


def say(msg):
    print(f"[build_chat {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, cwd, env, quiet=True):
    p = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.DEVNULL if quiet else None, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        sys.exit(f"STOP: {' '.join(map(str, cmd))} (in {cwd}) failed:\n{p.stderr[-2000:]}")


def size_mb(p):
    total = 0
    for d, _, files in os.walk(p):
        for f in files:
            try:
                total += os.lstat(os.path.join(d, f)).st_size
            except OSError:
                pass
    return round(total / 1e6, 1)


def make_pictures(out_dir):
    """Pictures from the mascot (cale/brand/carrot.svg): the app icon (sky tile), the carrot, and wide logos (carrot plus the name).
    cale/host/mac/MascotIcon.swift draws them on a Mac. Elsewhere the ready-made cale/brand/carrot.png is used."""
    brand = ROOT / "cale" / "brand"
    svg = brand / "carrot.svg"
    tool = out_dir / "MascotIcon"

    def draw(size, name, bg=None):
        dest = out_dir / name
        if tool.exists():
            subprocess.run([str(tool), str(svg), str(dest), str(size)] + ([bg] if bg else []), check=True)
        else:
            try:
                from PIL import Image
                Image.open(brand / "carrot.png").convert("RGBA").resize((size, size), Image.LANCZOS).save(dest)
            except ImportError:
                shutil.copyfile(brand / "carrot.png", dest)
        return dest

    if sys.platform == "darwin" and shutil.which("swiftc"):
        subprocess.run(["swiftc", "-O", "-o", str(tool), str(ROOT / "cale" / "host" / "mac" / "MascotIcon.swift")], check=True, capture_output=True)
    icon = draw(256, "icon.png", "8fd3f4")
    carrot = draw(512, "carrot.png")
    wide, wide_dark = out_dir / "wordmark.png", out_dir / "wordmark-on-dark.png"
    glyph = draw(91, "glyph.png")
    try:
        from PIL import Image, ImageDraw, ImageFont
        h, w = 99, 704
        font = None
        for f in ("/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf", "/Library/Fonts/Arial Rounded Bold.ttf", "C:/Windows/Fonts/ARLRDBD.TTF",
                  "/System/Library/Fonts/Helvetica.ttc"):
            try:
                font = ImageFont.truetype(f, 66)
                break
            except OSError:
                continue
        g = Image.open(glyph).convert("RGBA")
        for path, color in ((wide, (59, 35, 20, 255)), (wide_dark, (255, 241, 214, 255))):
            canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            canvas.paste(g, (0, 4), g)
            if font:
                ImageDraw.Draw(canvas).text((h + 4, h // 2 + 2), "Cale", font=font, fill=color, anchor="lm")
            canvas.save(path)
    except ImportError:
        for path in (wide, wide_dark):               # no Pillow: the carrot alone
            shutil.copyfile(glyph, path)
    return dict(carrot_svg=svg, carrot_png=carrot, icon=icon, wordmark=wide, wordmark_on_dark=wide_dark)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--node-dir", required=True)
    ap.add_argument("--yarn")
    ap.add_argument("--out", default=str(ROOT / "dist" / "chat"))
    ap.add_argument("--work")
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()
    src, out = Path(a.src).resolve(), Path(a.out).resolve()
    work = Path(a.work).resolve() if a.work else out.with_name(out.name + ".work")
    node_bin = Path(a.node_dir).resolve() / "bin"
    if not (node_bin / "node").exists():
        sys.exit(f"STOP: no node at {node_bin / 'node'}")
    for sub in ("server", "collector", "frontend"):
        if not (src / sub).is_dir():
            sys.exit(f"STOP: {src / sub} is missing. --src must be an AnythingLLM checkout.")

    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    work.mkdir(parents=True)
    out.mkdir(parents=True)
    env = dict(os.environ, PATH=f"{node_bin}{os.pathsep}{os.environ['PATH']}", PUPPETEER_SKIP_DOWNLOAD="1", PUPPETEER_SKIP_CHROMIUM_DOWNLOAD="1",
               CHECKPOINT_DISABLE="1", PRISMA_HIDE_UPDATE_MESSAGE="1", npm_config_update_notifier="false", NODE_ENV="")
    yarn = a.yarn or shutil.which("yarn")
    if not yarn:
        say("installing yarn 1 (build tool only) next to the work folder")
        run(["npm", "install", "--silent", "--no-audit", "--no-fund", "--prefix", str(work / "yarnbox"), "yarn@1.22.22"], work, env)
        yarn = str(work / "yarnbox" / "node_modules" / ".bin" / "yarn")
    yarn_install = [yarn, "install", "--frozen-lockfile", "--silent", "--ignore-engines", "--non-interactive"]

    for sub in ("server", "collector", "frontend"):
        shutil.copytree(src / sub, work / sub, ignore=SKIP)
    shutil.copyfile(src / "LICENSE", out / "LICENSE-AnythingLLM.txt")

    # 1. frontend: rebrand the source, then build
    say("pictures and rebrand")
    pictures = make_pictures(work)
    from cale.anythingllm import rebrand
    rebrand.apply(work / "frontend", **pictures)
    say("frontend: install and build")
    run(yarn_install, work / "frontend", env)
    run([yarn, "build"], work / "frontend", env)

    # 2. server: full install (the Prisma tool is needed once), make the empty database, then keep only production packages
    say("server: install (with the build tools)")
    run(yarn_install, work / "server", env)
    say("server: Prisma client and the empty database")
    srv = work / "server"
    (srv / "storage").mkdir(exist_ok=True)
    run([str(srv / "node_modules" / ".bin" / "prisma"), "generate", "--schema=./prisma/schema.prisma"], srv, env)
    run([str(srv / "node_modules" / ".bin" / "prisma"), "migrate", "deploy", "--schema=./prisma/schema.prisma"], srv, env)
    shutil.copyfile(srv / "storage" / "anythingllm.db", out / "template.db")
    prisma_gen = work / "prisma-generated"
    shutil.copytree(srv / "node_modules" / ".prisma", prisma_gen)                  # holds the query engine (a file). No download at run time.
    shutil.rmtree(srv / "node_modules")
    shutil.rmtree(srv / "storage")
    say("server: install (production packages only)")
    run(yarn_install + ["--production"], srv, env)
    shutil.rmtree(srv / "node_modules" / ".prisma", ignore_errors=True)
    shutil.copytree(prisma_gen, srv / "node_modules" / ".prisma")

    # 3. collector without Chromium
    say("collector: install (production packages only, no Chromium)")
    run(yarn_install + ["--production"], work / "collector", env)

    # 4. the embedding file, so the first document needs no internet
    say("embedding file")
    seed = out / "models-seed"
    seed.mkdir()
    fetch = ('import("@xenova/transformers").then(async ({pipeline,env})=>{env.cacheDir=process.argv[1];'
             f'const p=await pipeline("feature-extraction","{EMBEDDER}",{{cache_dir:process.argv[1]}});'
             'const r=await p("hello",{pooling:"mean",normalize:true});console.log("embedder ok",r.data.length)})')
    run([str(node_bin / "node"), "-e", fetch, str(seed)], srv, env, quiet=False)

    # 5. put the bundle together
    say("bundle")
    shutil.copytree(srv, out / "server", symlinks=True, ignore=top_only("storage"))
    shutil.copytree(work / "frontend" / "dist", out / "server" / "public")
    meta = out / "server" / "utils" / "boot" / "MetaGenerator.js"      # the page's title and share tags: name Cale, no link to the AnythingLLM site
    text = meta.read_text()
    for old, new in (("AnythingLLM | Your personal LLM trained on anything", "Cale"), ("https://anythingllm.com", ""),
                     ("AnythingLLM", "Cale")):
        text = text.replace(old, new)
    text = re.sub(r"https://raw\.githubusercontent\.com/Mintplex-Labs/[^\"]*", "", text)
    meta.write_text(text)
    shutil.copytree(work / "collector", out / "collector", symlinks=True, ignore=top_only("hotdir"))
    shutil.copyfile(ROOT / "cale" / "anythingllm" / "preload.js", out / "preload.js")
    prune(out)
    engines = sorted(p.name for p in (out / "server" / "node_modules" / ".prisma" / "client").glob("libquery_engine*"))
    (out / "bundle.json").write_text(json.dumps({"anythingllm": json.loads((src / "server" / "package.json").read_text())["version"], "prisma_engines": engines, "embedder": EMBEDDER,
        "node": subprocess.run([str(node_bin / "node"), "--version"], capture_output=True, text=True).stdout.strip(),
        "built": time.strftime("%Y-%m-%d")}, indent=1))
    if not a.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    say(f"DONE {out}  {size_mb(out)} MB   (server {size_mb(out / 'server')}, collector {size_mb(out / 'collector')}, "
        f"models-seed {size_mb(seed)}, template.db {size_mb(out / 'template.db')})")


if __name__ == "__main__":
    main()
