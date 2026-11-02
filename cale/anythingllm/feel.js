// The chat's "Quicker <-> Smarter" slider. Loaded by preload.js (same process as the chat server).
// The page asks /cale-feel. This file turns a slider position into a speed goal for the chat's model object in Cale.
// Cale then picks the size that meets the goal on THIS computer, so the same position means the same feel everywhere.
// The ends of the slider come from this computer: the quicker end is the fastest size that is allowed to run here,
// the smarter end is the largest. A weaker computer's slider covers smaller sizes. The slider looks the same.
// Nothing here names a size or a number to the person.
// Choice (2026-10-18): the slider reaches the largest size that is allowed to run here, even when that size is a little slower than reading
// speed (the HP's largest size runs at about 7 tokens/s). A person may choose that. The DEFAULT position never does: it sits at the largest
// size that still reads as flowing text (defaultPosition). So the default is the same feel everywhere, and the ends are the person's choice.
"use strict";
const http = require("http");
const fs = require("fs");

// POLICY (docs/CONSTANTS.md): the slowest text flow that still reads as flowing. Average adult reading is about 5 tokens/s.
// Below this, text arrives slower than people read.
const READING_TPS = 7;
// A speed goal exactly at a size's predicted speed would be missed by a small error. These keep the ends inside what the sizes reach.
const QUICK_MARGIN = 0.98;       // the quicker end asks for 98% of the fastest allowed size's speed
const SMART_MARGIN = 0.9;        // the smarter end asks for 90% of the largest allowed size's speed
const SYNC_MS = 30000;           // how often the goal is set again from the saved position (Cale may have restarted or learned new speeds)

/** Which sizes may run here, and what the two ends of the slider are. `info` is cale.this_computer from GET /v1/models.
 *  memoryPct is the share of the computer's memory the chat may use (the chat's own memory goal). Returns null when Cale cannot say yet. */
function ends(info, memoryPct) {
  if (!info || !info.sizes || !(info.ram_gb > 0)) return null;
  const all = Object.values(info.sizes).filter((s) => s && s.decode_tps > 0 && s.needs_gb > 0);
  if (!all.length) return null;
  const cap = memoryPct > 0 ? (memoryPct / 100) * info.ram_gb : Infinity;
  let ok = all.filter((s) => s.needs_gb <= cap);
  if (!ok.length) ok = [all.reduce((a, b) => (b.needs_gb < a.needs_gb ? b : a))];       // nothing fits: the smallest one is all there is
  const biggest = Math.max(...ok.map((s) => s.size_pct));
  const smart = Math.min(...ok.filter((s) => s.size_pct === biggest).map((s) => s.decode_tps));
  const quick = Math.max(...ok.map((s) => s.decode_tps));
  const lo = SMART_MARGIN * smart;
  const hi = Math.max(lo, QUICK_MARGIN * quick);
  return { lo, hi };
}

/** Slider position (0 = quicker, 1 = smarter) to a speed goal (tokens per second). */
function goalFor(position, e) {
  const p = Math.min(1, Math.max(0, position));
  return e.lo + (1 - p) * (e.hi - e.lo);
}

/** Where the slider sits when the person has not moved it: the largest size that still reads as flowing text.
 *  If even the largest size is faster than reading speed, that is the smarter end. */
function defaultPosition(e) {
  if (e.hi <= e.lo) return 1;
  const goal = Math.max(READING_TPS, e.lo);
  return Math.min(1, Math.max(0, 1 - (goal - e.lo) / (e.hi - e.lo)));
}

// ---- the part that talks to Cale and to the page ------------------------------------------------------------------------
const BASE = (process.env.GENERIC_OPEN_AI_BASE_PATH || "").replace(/\/v1\/?$/, "");     // http://127.0.0.1:47821
const KEY = process.env.GENERIC_OPEN_AI_API_KEY || "";
const OBJECT = process.env.GENERIC_OPEN_AI_MODEL_PREF || "chat";
const FILE = process.env.CALE_FEEL_FILE || "";
let applied = null;                                            // the goal last sent to Cale

function readSaved() {
  try {
    const p = JSON.parse(fs.readFileSync(FILE, "utf8")).position;
    return typeof p === "number" && p >= 0 && p <= 1 ? p : null;
  } catch { return null; }
}
function writeSaved(position) {
  if (!FILE) return;
  try { fs.writeFileSync(FILE, JSON.stringify({ position })); } catch { /* the slider still works for this session */ }
}

function cale(method, path, body) {
  return new Promise((resolve, reject) => {
    let u;
    try { u = new URL(BASE + path); } catch (e) { return reject(e); }
    const data = body === undefined ? null : Buffer.from(JSON.stringify(body));
    const req = http.request({ host: u.hostname, port: u.port, path: u.pathname, method, timeout: 8000,
      headers: { Authorization: `Bearer ${KEY}`, ...(data ? { "Content-Type": "application/json", "Content-Length": data.length } : {}) } }, (res) => {
      const chunks = [];
      res.on("data", (c) => chunks.push(c));
      res.on("end", () => {
        let json = null;
        try { json = JSON.parse(Buffer.concat(chunks).toString("utf8")); } catch { /* not json */ }
        resolve({ status: res.statusCode, json });
      });
    });
    req.on("timeout", () => req.destroy(new Error("timeout")));
    req.on("error", reject);
    if (data) req.write(data);
    req.end();
  });
}

/** What this computer can do for the chat's object right now, or null. */
async function look() {
  const r = await cale("GET", "/v1/models");
  if (r.status !== 200 || !r.json) return null;
  const mine = (r.json.data || []).find((d) => d.id === OBJECT);
  const e = ends(r.json.cale && r.json.cale.this_computer, mine && mine.cale && mine.cale.spec && mine.cale.spec.memory_pct);
  return mine && e ? { e, spec: mine.cale.spec } : null;
}

/** Set Cale's goal from a position. Returns the position used, or null when Cale is not ready. */
async function apply(position) {
  const s = await look();
  if (!s) return null;
  const pos = position === null || position === undefined ? defaultPosition(s.e) : position;
  const goal = Math.round(goalFor(pos, s.e) * 100) / 100;
  const now = s.spec && s.spec.min_tokens_per_s;
  if (applied === goal && now === goal) return pos;
  const r = await cale("POST", `/cale/models/${encodeURIComponent(OBJECT)}`, { min_tokens_per_s: goal });
  if (r.status !== 200) return null;
  applied = goal;
  return pos;
}

async function state() {
  try {
    const s = await look();
    if (!s) return { ready: false };
    const saved = readSaved();
    return { ready: true, position: saved === null ? defaultPosition(s.e) : saved, saved: saved !== null };
  } catch { return { ready: false }; }
}

function send(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(body), "Cache-Control": "no-store" });
  res.end(body);
}

/** The page's calls: GET /cale-feel and POST /cale-feel {"position": 0..1}. preload.js has already checked the host, origin and cookie. */
function handle(req, res) {
  if (req.method === "GET") {
    state().then((s) => send(res, 200, s));
    return;
  }
  if (req.method !== "POST") return send(res, 405, { error: "method" });
  const chunks = []; let size = 0;
  req.on("data", (c) => { size += c.length; if (size > 1024) { req.destroy(); } else chunks.push(c); });
  req.on("end", () => {
    let p;
    try { p = JSON.parse(Buffer.concat(chunks).toString("utf8")).position; } catch { return send(res, 400, { error: "body" }); }
    if (typeof p !== "number" || !(p >= 0 && p <= 1)) return send(res, 400, { error: "position must be a number from 0 to 1" });
    writeSaved(p);
    apply(p).then((used) => send(res, used === null ? 503 : 200, { ready: used !== null, position: p })).catch(() => send(res, 503, { ready: false }));
  });
}

// Keep Cale in line with the slider: after a restart of Cale (the object is made again with its first goal), and when Cale learns new speeds.
// Until Cale can say what this computer does (the AI is still loading), look again every few seconds. Then every SYNC_MS.
if (FILE && BASE && KEY) {
  const tick = (wait) => {
    const t = setTimeout(() => {
      apply(readSaved()).then((used) => tick(used === null ? 3000 : SYNC_MS), () => tick(3000));
    }, wait);
    if (t.unref) t.unref();
  };
  tick(3000);
}

module.exports = { handle, ends, goalFor, defaultPosition, READING_TPS };
