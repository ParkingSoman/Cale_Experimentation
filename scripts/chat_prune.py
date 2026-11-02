"""Remove the files of the chat bundle that nothing can reach, after the server and the document reader were packed (scripts/pack_chat.mjs).

Why: the chat is about 61,000 files. Starting it opened about 2,600 of them. The rest are never used by the code that is packed, but on a laptop with a
virus scanner every file costs time when it is unpacked and copied (209 s to unpack on a 4-core Windows laptop), and the download carries them.

What is kept, whatever the packer saw:
  1. every package that must stay a separate file (the `external` list of the pack: native add-ons, the embedding library), with everything they depend on;
  2. every package that has code which loads a file whose name is only known when the program runs (a computed require), and its dependencies;
  3. every package named in the unpacked project files (the document converters load their libraries from disk), and in project files with computed requires;
  4. every file that is not JavaScript or a document (pictures, fonts, character maps, native add-ons, data): code may read them next to itself;
  5. every package.json;
  6. the whole project source (server/ and collector/ outside node_modules).
What is removed: the JavaScript and documentation files of every other package. Their license texts go into ONE file, THIRD-PARTY-LICENSES.txt.
usage: python chat_prune.py BUNDLE_FOLDER META_server.json META_collector.json
"""
import fnmatch
import json
import os
import re
import sys
from pathlib import Path

REMOVE_EXT = {".js", ".cjs", ".mjs", ".jsx", ".ts", ".mts", ".cts", ".tsx", ".map", ".md", ".markdown", ".flow"}          # not .html, .txt, .yml: code may read them as templates or data
LICENSE_RE = re.compile(r"^(licen[sc]e|notice|copying|patents)(\.|-|$)", re.I)
KEEP_NAMES = {"package.json"}


def read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def packages(root):
    """Every package folder under root/node_modules (nested ones too): {folder: name}."""
    found = {}
    for d, dirs, files in os.walk(root / "node_modules"):
        if "package.json" in files and Path(d) != root / "node_modules":
            parent = Path(d).parent
            scoped = parent.name.startswith(("@", ".")) and parent.parent.name == "node_modules"
            if parent.name != "node_modules" and not scoped:
                continue                                    # a marker file inside a package ({"type": "commonjs"} in dist/cjs): not a package of its own
            found[Path(d)] = (parent.name + "/" + Path(d).name) if scoped else Path(d).name      # @scope/name, and .prisma/client
    return found


def resolve_dep(dep, frm, root):
    """Where node would find package `dep` when it is required from folder `frm` (nearest node_modules going up)."""
    cur = frm
    while True:
        cand = cur / "node_modules" / dep
        if (cand / "package.json").exists():
            return cand
        if cur == root or cur.parent == cur:
            return None
        cur = cur.parent


def owner(path, pkgs):
    """The package folder a file belongs to (the nearest one above it)."""
    cur = Path(path).parent
    while True:
        if cur in pkgs:
            return cur
        if cur.parent == cur:
            return None
        cur = cur.parent


def project_files(root):
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x != "node_modules"]
        for f in files:
            if f.endswith((".js", ".cjs", ".mjs")):
                yield Path(d) / f


BARE = re.compile(r"""(?:require|import)\s*\(\s*['"]([^'"./][^'"]*)['"]\s*\)|\bfrom\s+['"]([^'"./][^'"]*)['"]|\brequire\.resolve\(\s*['"]([^'"./][^'"]*)['"]""")


# POLICY: project files the Cale chat never starts. A Telegram bot has no screen in Cale's chat, and test and sample scripts are run by hand. Their
# packages are not kept for them. (If one of them is ever started, it fails with "cannot find module", which is logged in the chat log.)
NEVER_STARTED = re.compile(r"telegram|test|[/\\]scripts?[/\\]|syncstaticlists|[/\\]cli\.js$|example|seed\.js")
REL = re.compile(r"""(?:require|import)\s*\(\s*['"](\.{1,2}/[^'"]*)['"]\s*\)|\bfrom\s+['"](\.{1,2}/[^'"]*)['"]""")


def pkg_name(spec):
    parts = spec.split("/")
    return "/".join(parts[:2]) if spec.startswith(("@", ".")) else parts[0]


def prune(out, part, meta):
    root = out / part
    pkgs = packages(root)
    by_name = {}
    for d, n in pkgs.items():
        by_name.setdefault(n, []).append(d)
    inputs = {Path(p) for p in meta["inputs"]}
    dynamic = {Path(p) for p in meta["dynamic"]}
    external = meta["external"]

    keep = set()

    def add(d):
        if d is not None and d not in keep:
            keep.add(d)
            pj = read_json(d / "package.json")
            for field in ("dependencies", "optionalDependencies", "peerDependencies"):
                for dep in (pj.get(field) or {}):
                    add(resolve_dep(dep, d, root))

    # 1. packages that stay separate files
    for n, ds in by_name.items():
        if any(fnmatch.fnmatch(n, pat) or n == pat for pat in external):
            for d in ds:
                add(d)
    # 2. packages with code that loads a file by a name known only at run time
    for f in dynamic:
        o = owner(f, pkgs)
        if o is not None:
            add(o)
    # 3. packages named by project files that are still loaded from disk as they were: the files the packer could not pack (meta["unpacked"]), and the
    #    project files with computed requires (they may name a package in a list: patchSdkTimeouts.js). The other project files are packed whole.
    outputs = {Path(p) for p in meta.get("outputs", [])}                 # the packs and the stubs: they name every package, but those are inside them
    # Files that can still be STARTED from disk (the converters and jobs that could not be packed): what they require must stay, with what those files load.
    unpacked = {Path(p) for p in meta.get("scan", meta.get("unpacked", [])) if not NEVER_STARTED.search(str(Path(p).relative_to(root)).lower()) and Path(p) not in outputs}
    sources = {f: f.read_text(encoding="utf-8", errors="replace") for f in unpacked if f.exists()}
    disk = set(sources)
    todo = list(disk)
    while todo:                                                     # what those load in turn through relative requires
        f = todo.pop()
        for m in REL.finditer(sources[f]):
            spec = m.group(1) or m.group(2)
            base = (f.parent / spec).resolve()
            for cand in (base, base.with_name(base.name + ".js"), base.with_name(base.name + ".cjs"), base / "index.js"):
                if cand.is_file() and cand not in disk and cand not in outputs and cand.suffix in (".js", ".cjs", ".mjs"):
                    sources[cand] = cand.read_text(encoding="utf-8", errors="replace")
                    disk.add(cand)
                    todo.append(cand)
                    break
    for f in disk:
        for m in BARE.finditer(sources[f]):
            spec = m.group(1) or m.group(2) or m.group(3)
            if spec:
                for d in by_name.get(pkg_name(spec), []):
                    add(d)
    # Project files inside a pack with a computed require (patchSdkTimeouts.js): they may name packages in a list, as strings. Only those names count:
    # what the file requires by name is inside the pack.
    for f in dynamic:
        if "node_modules" in f.relative_to(root).parts or not f.exists():
            continue
        for q in re.findall(r"""['"]((?:@[\w.-]+/)?[\w.-]+)['"]""", f.read_text(encoding="utf-8", errors="replace")):
            for d in by_name.get(q, []):
                add(d)

    if os.environ.get("CHAT_PRUNE_EXPLAIN") == "1":                    # which seed keeps how many packages (set CHAT_PRUNE_EXPLAIN=1)
        def closure_of(seed_dirs):
            seen = set()

            def go(d):
                if d is not None and d not in seen:
                    seen.add(d)
                    pj = read_json(d / "package.json")
                    for field in ("dependencies", "optionalDependencies", "peerDependencies"):
                        for dep in (pj.get(field) or {}):
                            go(resolve_dep(dep, d, root))
            for d in seed_dirs:
                go(d)
            return seen
        ext = [d for n, ds in by_name.items() if any(fnmatch.fnmatch(n, pat) or n == pat for pat in external) for d in ds]
        print(f"  [explain {part}] kept {len(keep)} of {len(pkgs)}; externals alone: {len(closure_of(ext))}")
        for f in sorted(disk):
            seeds = [d for m in BARE.finditer(sources[f]) for spec in [m.group(1) or m.group(2) or m.group(3)] if spec for d in by_name.get(pkg_name(spec), [])]
            n = len(closure_of(seeds))
            if n > 5:
                print(f"  [explain {part}] {str(f.relative_to(root)):70s} brings {n} packages by itself")

    # remove, and collect the licenses of what goes
    licenses, removed_files, kept_files = [], 0, 0
    for d, name in sorted(pkgs.items(), key=lambda kv: str(kv[0])):
        if d in keep:
            continue
        pj = read_json(d / "package.json")
        texts = []
        for dirpath, dirs, files in os.walk(d):
            dirs[:] = [x for x in dirs if x != "node_modules"]                 # a nested package is judged on its own
            for f in files:
                p = Path(dirpath) / f
                if LICENSE_RE.match(f):
                    try:
                        texts.append((f, p.read_text(encoding="utf-8", errors="replace")))
                    except OSError:
                        pass
                    p.unlink()
                    removed_files += 1
                elif f in KEEP_NAMES:
                    kept_files += 1
                elif "worker" in str(p.relative_to(d)).lower() or f.endswith(".min.js"):
                    kept_files += 1                                          # a worker script may be started by its path; a .min.js may be read as data
                elif p.suffix.lower() in REMOVE_EXT or f.endswith(".d.ts"):
                    p.unlink()
                    removed_files += 1
                else:
                    kept_files += 1
        if texts:
            lic = pj.get("license")
            if not lic and isinstance(pj.get("licenses"), list) and pj["licenses"]:
                lic = pj["licenses"][0].get("type") if isinstance(pj["licenses"][0], dict) else pj["licenses"][0]
            licenses.append(f"==== {name}@{pj.get('version', '?')} ({lic if isinstance(lic, str) else json.dumps(lic)}) ====\n" + "\n\n".join(t for _, t in texts).strip() + "\n")
    # empty folders
    for d, dirs, files in os.walk(root / "node_modules", topdown=False):
        if not dirs and not files:
            try:
                os.rmdir(d)
            except OSError:
                pass
    return {"packages": len(pkgs), "kept_packages": len(keep), "removed_files": removed_files, "other_files_left": kept_files}, licenses


def main():
    out = Path(sys.argv[1])
    all_licenses = []
    for part, meta_file in zip(("server", "collector"), sys.argv[2:4]):
        meta = read_json(meta_file)
        stat, lic = prune(out, part, meta)
        all_licenses += lic
        print(f"pruned {part}: {stat}")
    (out / "THIRD-PARTY-LICENSES.txt").write_text(
        "The chat window program contains the open source packages below. Their code is packed into server/index.js and collector/index.js, or removed\n"
        "after packing. Their license texts are here.\n\n" + "\n".join(all_licenses), encoding="utf-8")
    print("licenses written:", len(all_licenses), "packages")


if __name__ == "__main__":
    main()
