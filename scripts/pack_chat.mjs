// Pack one Node program (AnythingLLM's server or its document reader) into ONE file, so Node opens one file at start instead of about 2,600.
// Why: starting the chat server took 1 s on a Mac and 4.7 s on a 4-core Windows laptop. The same code, but the laptop pays for every file it opens
// (disk and virus scanner). Measured with scripts/chat_race.py (docs/HANDOFF-2026-10-18.md).
//
// usage: node pack_chat.mjs ROOT ENTRY OUT [external ...]      (run in a folder where `npm install esbuild` was done)
//   ROOT   the program's folder (server/ or collector/). ENTRY and OUT are relative to it. ENTRY is read, OUT is written.
//   external  more package names to leave out of the pack (packages that are not there, or must stay files)
// What stays a separate file: native add-ons (sharp, onnxruntime-node, canvas, lancedb, prisma) and the embedding library (it loads its own pieces).
// Nothing is deleted: every original file stays on disk. A file that asks where it lives (__dirname, __filename) or that loads a file whose name is
// only known when the program runs (require(name)) is pointed at its original folder.
import { build } from "esbuild";
import fs from "node:fs";
import path from "node:path";

const [root, entry, out, ...more] = process.argv.slice(2);
const external = ["sharp", "onnxruntime-node", "canvas", "@lancedb/*", "@prisma/*", ".prisma/*", "@mapbox/node-pre-gyp", "@xenova/transformers", "@xenova/transformers/*", ...more];

const dynamicFiles = new Set();                                    // files that load a file whose name is only known at run time
const keepPlaces = {
  name: "keep-places",
  setup(b) {
    b.onLoad({ filter: /\.(c|m)?js$/ }, async (a) => {
      let text = await fs.promises.readFile(a.path, "utf8");
      const uses = (n) => new RegExp("\\b" + n + "\\b").test(text);
      const declares = (n) => new RegExp("\\b(?:const|let|var|function|class)\\s+" + n + "\\b|\\bimport\\s+[^;]*\\b" + n + "\\b").test(text);
      const rel = path.relative(root, a.path).split(path.sep).join("/");
      const relDir = path.posix.dirname(rel);
      const decls = [];
      if (uses("__filename") && !declares("__filename")) decls.push(`__filename = require("path").join(__CALE_ROOT, ${JSON.stringify(rel)})`);
      if (uses("__dirname") && !declares("__dirname")) decls.push(`__dirname = require("path").join(__CALE_ROOT, ${JSON.stringify(relDir)})`);
      // A computed require in a PROJECT file (outside node_modules) is pointed at the folder of that file: the document reader picks its converter that way, and
      // the converters stay files. A computed require inside a LIBRARY is left as esbuild makes it: the usual paths are packed (express, body-parser) and an
      // unusual one (a view engine, a database driver Cale does not use) fails only if it runs. Pointing those at files would keep every such library on disk.
      const isProject = !a.path.split(path.sep).includes("node_modules");
      const computed = /(?<![\w.$])require\(\s*(?![`"'])/g;
      const hasComputed = isProject && computed.test(text);
      computed.lastIndex = 0;
      if (hasComputed) { text = text.replace(computed, `__CALE_REQ(${JSON.stringify(relDir)})(`); dynamicFiles.add(a.path); }
      if (!decls.length && !hasComputed) return null;
      const head = decls.length ? `var ${decls.join(", ")};\n` : "";
      if (text.startsWith("#!")) { const i = text.indexOf("\n") + 1; text = text.slice(0, i) + head + text.slice(i); } else text = head + text;
      return { contents: text, loader: "js", resolveDir: path.dirname(a.path) };
    });
  },
};

const bannerFor = (relDir) => {
  const up = relDir === "." ? "" : relDir.split("/").map(() => "..").join("/");        // from the folder of the packed file up to the program's folder
  return `var __CALE_ROOT = require("path").join(__dirname, ${JSON.stringify(up)}); var __CALE_REQ = (d) => require("module").createRequire(require("path").join(__CALE_ROOT, d, "x.js"));`;
};
const common = { bundle: true, platform: "node", target: "node22", legalComments: "none" };
const autoExternal = new Set();                                  // libraries that cannot be packed (see below); they stay files and are kept by chat_prune.py
const pkgOf = (file) => {
  const parts = file.split(path.sep), i = parts.lastIndexOf("node_modules");
  if (i < 0 || i + 1 >= parts.length) return null;
  return parts[i + 1].startsWith("@") ? parts[i + 1] + "/" + parts[i + 2] : parts[i + 1];
};
// A library that asks for `import.meta` (where its own file is) does not work inside a pack: esbuild says so in a warning. Such a library stays a separate
// file. The build is repeated with it left out until no warning is left.
async function buildOne(options) {
  for (let round = 0; round < 8; round++) {
    const r = await build({ ...common, ...options, logLevel: "silent", plugins: [keepPlaces],
      external: [...external, ...[...autoExternal].flatMap((p) => [p, p + "/*"])] });
    const more = new Set();
    for (const w of r.warnings) {
      if (/import\.meta/.test(w.text) && w.location && w.location.file) {
        const pkg = pkgOf(path.resolve(process.cwd(), w.location.file));
        if (pkg && !autoExternal.has(pkg)) more.add(pkg);
      }
    }
    if (!more.size) return r;
    more.forEach((p) => autoExternal.add(p));
  }
  throw new Error("too many rounds of libraries that cannot be packed");
}

// 1. the program itself, from its start file
const res = await buildOne({ metafile: true, entryPoints: [path.join(root, entry)], outfile: path.resolve(root, out), banner: { js: bannerFor(path.posix.dirname(out)) } });
const packed = new Set(Object.keys(res.metafile.inputs).map((f) => path.resolve(process.cwd(), f)));

// 2. (the document reader only: PACK_STAGE2=1) every other project file that the program loads from disk by a name that is only known when it runs
//    (the document converters): each becomes its own self-contained file at the same place, so that the libraries they use need not stay on disk.
//    The server's background jobs are not packed: each would carry a copy of the whole server (35 MB each).
const SKIP_DIR = new Set(["node_modules", "public", "example", "examples", "test", "tests", "__tests__", "swagger", "prisma"]);
const rest = [], unpacked = [];
(function walk(dir) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const f = path.join(dir, e.name);
    if (e.isDirectory()) { if (!SKIP_DIR.has(e.name)) walk(f); continue; }
    if (!/\.(c|m)?js$/.test(e.name) || /^eslint\.config/.test(e.name) || f === path.resolve(root, entry) || f === path.resolve(root, out)) continue;
    if (!packed.has(f)) rest.push(f);
  }
})(root);
const made = [];                                                    // every pack is built from the ORIGINAL files, and put in place only after all are built (a file packed first would be read again, packed, by the next)
for (const f of rest) {
  if (process.env.PACK_STAGE2 !== "1") { unpacked.push(f); continue; }
  const rel = path.relative(root, f).split(path.sep).join("/");
  const tmp = f + ".packing";
  try {
    await buildOne({ entryPoints: [f], outfile: tmp, banner: { js: bannerFor(path.posix.dirname(rel)) } });
    made.push([tmp, f]);
  } catch (e) {
    try { fs.unlinkSync(tmp); } catch {}
    unpacked.push(f);                                               // it stays as it was: the packages it needs must stay too (chat_prune.py)
  }
}
for (const [tmp, f] of made) fs.renameSync(tmp, f);
// 3. (the server only: PACK_JOBS=1) the background jobs (jobs/*.js) are started by name as workers, each a file of its own that loads a large part of the server.
//    Packed one by one, each would carry its own 30 MB copy of it. Instead ONE pack holds all of them behind a table of loaders (nothing runs until a loader is
//    called, so the code of a job runs exactly when its file used to), and each job file becomes a one-line stub that calls its loader.
const outputs = [path.resolve(root, out), ...made.map(([, f]) => f)];
if (process.env.PACK_JOBS === "1") {
  const dir = path.join(root, "jobs");
  const names = fs.existsSync(dir) ? fs.readdirSync(dir).filter((n) => /\.js$/.test(n) && !/^_/.test(n) && unpacked.includes(path.join(dir, n))) : [];
  if (names.length) {
    const entryFile = path.join(dir, "_all.entry.js");
    fs.writeFileSync(entryFile, "module.exports = {\n" + names.map((n) => `  ${JSON.stringify(n.replace(/\.js$/, ""))}: () => require(${JSON.stringify("./" + n)}),`).join("\n") + "\n};\n");
    try {
      await buildOne({ entryPoints: [entryFile], outfile: path.join(dir, "_all.js"), banner: { js: bannerFor("jobs") } });
      for (const n of names) {
        fs.writeFileSync(path.join(dir, n), `// Cale: this job's code is in _all.js (scripts/pack_chat.mjs)\nrequire("./_all.js")[${JSON.stringify(n.replace(/\.js$/, ""))}]();\n`);
        const i = unpacked.indexOf(path.join(dir, n));
        if (i >= 0) unpacked.splice(i, 1);
        outputs.push(path.join(dir, n));
      }
      outputs.push(path.join(dir, "_all.js"));
    } catch (e) {
      console.error("the jobs could not be packed together; they stay as files:", String(e.message || e).slice(0, 300));
    } finally {
      try { fs.unlinkSync(entryFile); } catch {}
    }
  }
}
// Files that can still be STARTED from disk by name, as they were (chat_prune.py keeps the packages they need): the converters and jobs that could not be packed.
// The other files that were not packed are not started by anything (every file the programs load is inside a pack, or is a converter or a job).
const jobsDir = path.join(root, "jobs");
const scan = unpacked.filter((f) => process.env.PACK_STAGE2 === "1" || (process.env.PACK_JOBS === "1" && path.dirname(f) === jobsDir) || process.env.PACK_JOBS !== "1");
const allExternal = [...external, ...[...autoExternal].flatMap((p) => [p, p + "/*"])];
// PACK_META: what the pack saw, for scripts/chat_prune.py.
if (process.env.PACK_META) {
  fs.writeFileSync(process.env.PACK_META, JSON.stringify({ inputs: [...packed], dynamic: [...dynamicFiles], external: allExternal, unpacked, scan, outputs }));
}
console.log("packed", path.join(path.basename(root), out), (fs.statSync(path.resolve(root, out)).size / 1e6).toFixed(1), "MB from", packed.size, "files; also packed", rest.length - unpacked.length, "more files", unpacked.length ? `(${unpacked.length} left as they were)` : "");
