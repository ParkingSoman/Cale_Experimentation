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
    <out>/feel.js            the Quicker/Smarter slider (loaded by preload.js)
    <out>/LICENSE-AnythingLLM.txt
    <out>/bundle.json        what is in it
    (server/index.js and collector/index.js are packed into one file each; the originals are index.orig.js. --no-pack skips this. Files that
     nothing can reach are then removed, and the license texts of their packages are in THIRD-PARTY-LICENSES.txt. --no-prune skips this.)

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


# Left out of the pack: packages that are not in the bundle on purpose (see UNUSED), and the browser tool (the document reader never opens web pages here).
PACK_EXTERNAL = ("../../package.json", "@azure/identity", "chromadb", "chromadb-default-embed", "puppeteer", "puppeteer-core")


def pack(work, node_bin, env, out, prune=False):
    """Pack the server and the document reader into one file each (scripts/pack_chat.mjs). Each starts by opening about 2,600 and 650 files; on a laptop with
    a virus scanner that took 4.7 s against 1 s on a Mac. The original index.js stays as index.orig.js. With prune=True, the files that nothing can reach
    are removed afterwards (scripts/chat_prune.py): about half of all files."""
    box = work / "packbox"
    box.mkdir(parents=True, exist_ok=True)
    run(["npm", "install", "--silent", "--no-audit", "--no-fund", "--prefix", str(box), "esbuild@0.24.0"], work, env)
    script = box / "pack_chat.mjs"
    shutil.copyfile(ROOT / "scripts" / "pack_chat.mjs", script)                  # next to node_modules, so that `import "esbuild"` finds it
    metas = []
    for part in ("server", "collector"):
        root = out / part
        (root / "index.js").rename(root / "index.orig.js")
        meta = box / f"meta-{part}.json"
        run([str(node_bin / "node"), str(script), str(root), "index.orig.js", "index.js", *PACK_EXTERNAL], box, dict(env, PACK_META=str(meta), PACK_STAGE2="1" if part == "collector" else "0", PACK_JOBS="1" if part == "server" else "0"), quiet=False)
        metas.append(meta)
    if prune:
        say("removing the files nothing can reach")
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "chat_prune.py"), str(out), *map(str, metas)], capture_output=True, text=True)
        print(r.stdout, end="")
        if r.returncode:
            sys.exit("STOP: chat_prune.py failed:\n" + r.stderr[-2000:])


OS_WORDS = ("darwin", "linux", "win32", "freebsd", "android")


def _foreign(name, osname, arch):
    """True if a native package or folder name (like lancedb-darwin-arm64, linuxmusl-x64, win32-x64-msvc) is for another system or chip.
    Only names that have a system word AND a chip word, so source folders such as lib/win32 are never touched."""
    n = name.lower()
    if not re.search(r"(^|[-_.])(%s)" % "|".join(OS_WORDS), n) or not re.search(r"(arm64|aarch64|x64|x86_64|ia32|arm|armv|s390x|ppc64|riscv64|loong64)", n):
        return False
    mine_os = {"win32": "win32", "darwin": "darwin", "linux": "linux"}[osname]
    chip = ("arm64", "aarch64") if arch == "arm64" else ("x64", "x86_64")
    return mine_os not in n or not any(c in n for c in chip)


def prune(bundle, osname=None, arch=None):
    """Remove what is not needed on this computer: unused packages, other systems' native files, maps, notes, type files."""
    import glob
    import platform
    osname = osname or {"darwin": "darwin", "win32": "win32"}.get(sys.platform, "linux")
    arch = arch or ("arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64")
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
            for d in list(dirs):                                              # other systems' or chips' native packages and folders
                if _foreign(d, osname, arch) and (Path(dirpath).name == "node_modules" or Path(dirpath).parent.name == "node_modules" and Path(dirpath).name.startswith("@")
                                                  or Path(dirpath).name in ("vendor", "napi-v3", "prebuilds") or Path(dirpath).parent.name == "vendor"):
                    shutil.rmtree(os.path.join(dirpath, d), ignore_errors=True)
                    dirs.remove(d)
            for f in files:
                if f.endswith((".map", ".d.ts", ".md")) and not f.upper().startswith("LICENSE"):
                    try:
                        os.remove(os.path.join(dirpath, f))
                    except OSError:
                        pass


def no_links(bundle):
    """Windows installers cannot carry symbolic links: remove the .bin links (tools for people, not used at run time), copy any other link's target."""
    for dirpath, dirs, files in os.walk(bundle):
        for name in dirs + files:
            p = Path(dirpath) / name
            if not p.is_symlink():
                continue
            target = p.resolve()
            p.unlink()
            if p.parent.name == ".bin" or not target.exists():
                continue
            shutil.copytree(target, p, symlinks=False) if target.is_dir() else shutil.copy2(target, p)
    for d in list(bundle.rglob(".bin")):
        shutil.rmtree(d, ignore_errors=True)


def shorten_for_windows(bundle):
    """Windows stops at 259 characters in a path (many programs; the person's user name is part of it). Remove what makes the longest paths
    and is never run: test, example and notes folders inside node_modules, and type-only files with a name over 100 characters."""
    for sub in ("server", "collector"):
        for dirpath, dirs, files in os.walk(bundle / sub / "node_modules"):
            for d in list(dirs):
                if d in ("test", "tests", "__tests__", ".github", "example", "examples", "docs") and Path(dirpath).name != "node_modules" and not Path(dirpath).name.startswith("@"):
                    shutil.rmtree(os.path.join(dirpath, d), ignore_errors=True)
                    dirs.remove(d)
            for f in files:
                if len(f) > 100 and f.endswith(".js"):
                    stem = f[:-3]
                    here = Path(dirpath)
                    users = [q for folder in (here, here.parent) for q in folder.glob("*.js") if q.name != f]
                    if not any(stem + '"' in q.read_text(errors="ignore") or stem + "'" in q.read_text(errors="ignore") for q in users):
                        os.remove(here / f)                              # nothing requires it (it only names a type)
    longest = max((len(str(p.relative_to(bundle))) for p in bundle.rglob("*") if p.is_file()), default=0)
    say(f"longest path inside the bundle: {longest} characters")
    return longest


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


def speed_patches(server):
    """Two costs at every start of the chat server that nothing needs until later. Each patch checks that the text it replaces is there, so a new
    AnythingLLM version stops the build instead of silently skipping the patch.
    1. comKey: a 2048-bit RSA key pair is made at every boot (93 ms on a fast computer, 7x that on a slow one) on the main thread. Now it is made on a
       helper thread. The collector is first called seconds later. If something signs before the pair is ready, the pair is made at once (and the
       late helper result is dropped, so the two files never come from different pairs).
    2. tiktoken: the token encoder is built when the first TokenManager is made. Now it is built once, 3 s after the start, in idle time (or on first use).
    """
    ck = server / "utils" / "comKey" / "index.js"
    t = ck.read_text()
    for old, new in (
        ("    if (generate) this.#generate();", "    if (generate) this.#generateLater();"),
        ("  #readPrivateKey() {\n", "  #readPrivateKey() {\n    if (CommunicationKey.pending) {\n      CommunicationKey.pending = false;\n      this.#generate();\n    }\n"),
        ("  #generate() {\n    const keyPair = crypto.generateKeyPairSync(\"rsa\", {\n      modulusLength: 2048,\n      publicKeyEncoding: {\n        type: \"pkcs1\",\n        format: \"pem\",\n      },\n      privateKeyEncoding: {\n        type: \"pkcs1\",\n        format: \"pem\",\n      },\n    });\n\n",
         "  #generateLater() {\n    CommunicationKey.pending = true;\n    crypto.generateKeyPair(\"rsa\", { modulusLength: 2048, publicKeyEncoding: { type: \"pkcs1\", format: \"pem\" }, privateKeyEncoding: { type: \"pkcs1\", format: \"pem\" } }, (err, publicKey, privateKey) => {\n      if (err || !CommunicationKey.pending) return;\n      CommunicationKey.pending = false;\n      this.#write({ publicKey, privateKey });\n    });\n  }\n\n  #generate() {\n    this.#write(crypto.generateKeyPairSync(\"rsa\", {\n      modulusLength: 2048,\n      publicKeyEncoding: { type: \"pkcs1\", format: \"pem\" },\n      privateKeyEncoding: { type: \"pkcs1\", format: \"pem\" },\n    }));\n  }\n\n  #write(keyPair) {\n"),
    ):
        assert old in t, f"comKey patch: text not found: {old[:50]!r}"
        t = t.replace(old, new, 1)
    t = t.replace("module.exports = { CommunicationKey };", "CommunicationKey.pending = false;\nmodule.exports = { CommunicationKey };")
    ck.write_text(t)
    tk = server / "utils" / "helpers" / "tiktoken.js"
    t = tk.read_text()
    old = "    this.encoder = getEncoding(this.encoderName);\n"
    assert old in t, "tiktoken patch: text not found"
    t = t.replace(old, "    const encoderName = this.encoderName;\n    Object.defineProperty(this, \"encoder\", { get: () => cachedEncoding(encoderName), configurable: true });\n", 1)
    # The table behind the encoder takes 190 ms to build on a slow laptop. Build it a few seconds after the start (the person is still typing), not in the first message.
    t = t.replace("class TokenManager {", "const encodings = {};\nconst cachedEncoding = (name) => (encodings[name] ||= getEncoding(name));\nsetTimeout(() => { try { cachedEncoding(\"cl100k_base\"); } catch (e) { /* built on first use instead */ } }, 3000).unref();\n\nclass TokenManager {", 1)
    assert "cachedEncoding(\"cl100k_base\")" in t, "tiktoken patch: class line not found"
    tk.write_text(t)
    # 3. Work at every start for parts the Cale chat does not use or does not need yet. The collector and the other providers are never reached in Cale.
    def edit(path, pairs, what):
        f = server / path
        text = f.read_text()
        for old_, new_ in pairs:
            assert old_ in text, f"{what}: text not found in {path}: {old_[:50]!r}"
            text = text.replace(old_, new_, 1)
        f.write_text(text)
    # the timeout patch loads the Anthropic SDK, which Cale's chat never calls (it talks to Cale through the OpenAI-style address)
    edit("utils/boot/patchSdkTimeouts.js", [('    ["@anthropic-ai/sdk", "Anthropic"],\n', "")], "sdk timeouts")
    # the Telegram library (a bot Cale's chat has no screen for) is loaded on first use
    edit("utils/telegramBot/index.js", [('const TelegramBot = require("node-telegram-bot-api");',
         'function TelegramBot(...a) {\n  return new (require("node-telegram-bot-api"))(...a);\n}')], "telegram")
    # the Chroma library (a vector database Cale does not use: it uses LanceDB) is loaded on first use
    edit("utils/vectorDbProviders/chroma/index.js", [('const { ChromaClient } = require("chromadb");',
         'function ChromaClient(...a) {\n  return new (require("chromadb").ChromaClient)(...a);\n}')], "chroma")
    edit("utils/vectorDbProviders/chromacloud/index.js", [('const { CloudClient } = require("chromadb");',
         'function CloudClient(...a) {\n  return new (require("chromadb").CloudClient)(...a);\n}')], "chroma cloud")
    # the encryption key (scrypt: 38 ms on a Mac, 130 ms on a slow laptop, twice at every start) is derived when it is first used
    edit("utils/EncryptionManager/index.js", [
        ("    this.key = crypto.scryptSync(this.#encryptionKey, this.#encryptionSalt, 32);\n",
         "    const keyText = this.#encryptionKey, saltText = this.#encryptionSalt;\n    let derived = null;\n    Object.defineProperty(this, \"key\", { get: () => (derived ||= crypto.scryptSync(keyText, saltText, 32)), configurable: true });\n"),
        ('    this.xPayload = this.key.toString("base64");\n',
         '    Object.defineProperty(this, "xPayload", { get: () => this.key.toString("base64"), configurable: true });\n')], "encryption")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--node-dir", required=True)
    ap.add_argument("--yarn")
    ap.add_argument("--out", default=str(ROOT / "dist" / "chat"))
    ap.add_argument("--work")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--no-prune", action="store_true", help="after packing, the files that nothing can reach are removed (scripts/chat_prune.py: about half of all files). This keeps them. After a change to the build, test with scripts/chat_race.py --docs --feel")
    ap.add_argument("--no-pack", action="store_true", help="do not pack the server and the document reader into one file each (slower to start)")
    ap.add_argument("--target", choices=("native", "win-x64"), default="native",
                    help="native: this computer (default). win-x64: a Windows bundle built on any computer (native files of the other system are fetched, not built)")
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
    cross = a.target == "win-x64"
    # Cross build: the full install (Prisma tool, embedding file) runs natively. The production install fetches the Windows files: yarn takes the
    # optional packages of every system (--ignore-platform), and sharp / canvas fetch their own files for npm_config_platform/arch.
    target_env = dict(env, npm_config_platform="win32", npm_config_arch="x64", npm_config_target_platform="win32", npm_config_target_arch="x64") if cross else env
    prod_install = yarn_install + ["--production"] + (["--ignore-platform"] if cross else [])

    out_seed = out / "models-seed"

    def fetch_embedder():
        """The embedding file, so the first document needs no internet. Runs with the packages of THIS computer."""
        say("embedding file")
        out_seed.mkdir()
        fetch = ('import("@xenova/transformers").then(async ({pipeline,env})=>{env.cacheDir=process.argv[1];'
                 f'const p=await pipeline("feature-extraction","{EMBEDDER}",{{cache_dir:process.argv[1]}});'
                 'const r=await p("hello",{pooling:"mean",normalize:true});console.log("embedder ok",r.data.length)})')
        run([str(node_bin / "node"), "-e", fetch, str(out_seed)], work / "server", env, quiet=False)

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
    say(f"recolor: {rebrand.recolor_built(work / 'frontend' / 'dist')} colours moved into the carrot palette")

    # 2. server: full install (the Prisma tool is needed once), make the empty database, then keep only production packages
    say("server: install (with the build tools)")
    run(yarn_install, work / "server", env)
    say("server: Prisma client and the empty database")
    srv = work / "server"
    (srv / "storage").mkdir(exist_ok=True)
    if cross:                                                                         # also make the Windows query engine (a file, fetched here)
        schema = srv / "prisma" / "schema.prisma"
        schema.write_text(schema.read_text().replace('provider = "prisma-client-js"', 'provider = "prisma-client-js"\n  binaryTargets = ["native", "windows"]', 1))
    run([str(srv / "node_modules" / ".bin" / "prisma"), "generate", "--schema=./prisma/schema.prisma"], srv, env)
    run([str(srv / "node_modules" / ".bin" / "prisma"), "migrate", "deploy", "--schema=./prisma/schema.prisma"], srv, env)
    shutil.copyfile(srv / "storage" / "anythingllm.db", out / "template.db")
    if cross:
        fetch_embedder()                                  # the full install here still has this computer's files; the Windows packages come next
    prisma_gen = work / "prisma-generated"
    shutil.copytree(srv / "node_modules" / ".prisma", prisma_gen)                  # holds the query engine (a file). No download at run time.
    shutil.rmtree(srv / "node_modules")
    shutil.rmtree(srv / "storage")
    say("server: install (production packages only)")
    run(prod_install, srv, target_env)
    shutil.rmtree(srv / "node_modules" / ".prisma", ignore_errors=True)
    shutil.copytree(prisma_gen, srv / "node_modules" / ".prisma")

    # 3. collector without Chromium
    say("collector: install (production packages only, no Chromium)")
    run(prod_install, work / "collector", target_env)

    if not cross:
        fetch_embedder()

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
    speed_patches(out / "server")
    shutil.copytree(work / "collector", out / "collector", symlinks=True, ignore=top_only("hotdir"))
    shutil.copyfile(ROOT / "cale" / "anythingllm" / "preload.js", out / "preload.js")
    shutil.copyfile(ROOT / "cale" / "anythingllm" / "feel.js", out / "feel.js")        # the Quicker/Smarter slider; preload.js loads it
    if cross:
        prune(out, "win32", "x64")
        for f in (out / "server" / "node_modules" / ".prisma" / "client").glob("libquery_engine*"):      # the engine of the build computer
            f.unlink()
        no_links(out)
        shorten_for_windows(out)
    else:
        prune(out)
    if not a.no_pack:
        say("packing the server and the document reader into one file each")
        pack(work, node_bin, env, out, prune=not a.no_prune)
    engines = sorted(p.name for p in (out / "server" / "node_modules" / ".prisma" / "client").glob("*query_engine*"))
    (out / "bundle.json").write_text(json.dumps({"anythingllm": json.loads((src / "server" / "package.json").read_text())["version"], "prisma_engines": engines, "embedder": EMBEDDER,
        "target": a.target, "packed": not a.no_pack, "pruned": not a.no_prune and not a.no_pack, "node": subprocess.run([str(node_bin / "node"), "--version"], capture_output=True, text=True).stdout.strip(),
        "built": time.strftime("%Y-%m-%d")}, indent=1))
    if not a.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    say(f"DONE {out}  {size_mb(out)} MB   (server {size_mb(out / 'server')}, collector {size_mb(out / 'collector')}, "
        f"models-seed {size_mb(out_seed)}, template.db {size_mb(out / 'template.db')})")


if __name__ == "__main__":
    main()
