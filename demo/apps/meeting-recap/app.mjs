// Meeting Recap: a tiny local server (node:http) that serves the page and calls Cale from Node.
// The browser never talks to Cale (Cale refuses requests that carry an Origin header).
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { connect, ensureRunning, CaleError } from "./cale-client.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PORT = 8824;
const CALE_URL = process.env.CALE_URL || "http://127.0.0.1:47821";
const KEY_FILE = path.join(HERE, "data", "cale-key");
const MAX_BODY = 1_000_000;

const SCHEMA = {
  type: "object",
  properties: {
    items: {
      type: "array",
      items: {
        type: "object",
        properties: { owner: { type: "string" }, task: { type: "string" }, due: { type: ["string", "null"] } },
        required: ["owner", "task", "due"],
        additionalProperties: false,
      },
    },
  },
  required: ["items"],
  additionalProperties: false,
};

// The lines between CALE-BEGIN and CALE-END markers below are the Cale calls. The Developer view shows them, read from this file.
let caleClient = null, recapObj = null, actionsObj = null, setupPromise = null, busy = false;

async function getCale() {
  if (process.env.CALE_ENSURE === "1") await ensureRunning(CALE_URL);
  let key = null;
  try { key = fs.readFileSync(KEY_FILE, "utf8").trim(); } catch { /* none yet */ }
  if (key) {
    const { Cale } = await import("./cale-client.mjs");
    const c = new Cale(CALE_URL, key);
    try { await c.models(); return c; }
    catch (e) { if (!(e instanceof CaleError) || (e.status !== 401 && e.status !== 403)) throw e; /* key rejected: ask again */ }
  }
  const c = await connect("Meeting Recap", { base: CALE_URL, onWait: (m) => console.error("Waiting for you to allow the app:", m) });
  fs.mkdirSync(path.dirname(KEY_FILE), { recursive: true });
  fs.writeFileSync(KEY_FILE, c.key, { mode: 0o600 });
  return c;
}

function setup() {
  if (!setupPromise) {
    setupPromise = (async () => {
      // CALE-BEGIN
      caleClient = await getCale();                       // connect("Meeting Recap"): the person allows it once, the key is kept
      recapObj = await caleClient.createModel({ name: "recap", min_tokens_per_s: 3, max_tokens: 400 });
      actionsObj = await caleClient.createModel({ name: "actions", max_first_token_s: 20, max_tokens: 600 });
      // CALE-END
    })().catch((e) => { setupPromise = null; throw e; });
  }
  return setupPromise;
}

const recapPrompt = (text) => [
  { role: "system", content: "You summarize meeting transcripts. Write 3 to 5 short bullet points, each starting with '- '. Plain text only, no bold, no markdown symbols other than the leading '- '. Cover the main plan, who does what and by when. Only use facts from the transcript. No intro, no closing line." },
  { role: "user", content: "Transcript:\n\n" + text + "\n\nWrite the recap now." },
];
const actionPrompt = (text) => [
  { role: "system", content: "You extract action items from meeting transcripts. Each item has: owner (first name of the person who must do it), task (short sentence), due (the deadline as written in the transcript, such as 'Thursday noon', or null if none). List EVERY task that a person committed to or was asked to do (including sending invites, reviews, tests, flags and emails), one item per task. Skip chatter such as lunch jokes. Use only facts from the transcript." },
  { role: "user", content: "Transcript:\n\n" + text + "\n\nList the action items." },
];

function plainError(e) {
  if (e instanceof CaleError) {
    if (e.code === "context_length_exceeded") return "This transcript is too long for this computer. Paste a shorter one.";
    if (e.status === 401 || e.status === 403) return "Cale did not accept this app. Delete data/cale-key and try again.";
    if (e.status === 507) return "This computer does not have enough free memory right now. Close some apps and try again.";
    if (e.status === 422) return "Cale cannot meet the speed goals on this computer right now.";
    if (e.status === 429) return "Cale is busy. Try again in a moment.";
    return "Cale could not answer: " + (e.body?.error?.message || "error " + e.status);
  }
  const m = String(e?.cause?.code || e?.code || "");
  if (m === "ECONNREFUSED" || /fetch failed/.test(String(e?.message))) return `Cale is not running at ${CALE_URL}. Start it with "cale start", then try again.`;
  if (/Nobody allowed/.test(String(e?.message))) return "Nobody allowed the app in time. Allow it in the Cale question, or type: cale allow \"Meeting Recap\"";
  return "Something went wrong: " + String(e?.message || e).slice(0, 200);
}

const round = (n, d = 1) => (typeof n === "number" ? Math.round(n * 10 ** d) / 10 ** d : null);

async function run(text, send, aborted) {
  const t0 = performance.now();
  send("status", { message: "Connecting to Cale..." });
  await setup();
  let size = null, first = null, recapText = "", meta = {};
  send("status", { message: "Writing the recap..." });
  // CALE-BEGIN
  // streamed recap: Cale picks the size that meets the speed goal
  const stream = caleClient.chatStream({ model: recapObj.id, messages: recapPrompt(text) });
  // CALE-END
  for await (const chunk of stream) {
    if (aborted()) return;
    if (chunk.cale) meta = { ...meta, ...chunk.cale };
    size = chunk.cale?.size ?? chunk.size ?? size;
    const word = chunk.choices?.[0]?.delta?.content;
    if (word) {
      if (first === null) first = (performance.now() - t0) / 1000;
      recapText += word;
      send("recap", { text: word });
    }
  }
  const recapDone = (performance.now() - t0) / 1000;
  send("recap_done", {});
  send("status", { message: "Finding action items..." });
  // CALE-BEGIN
  // action items: Cale checks every token, so the JSON always fits the schema
  const reply = await caleClient.chat({
    model: actionsObj.id, messages: actionPrompt(text),
    response_format: { type: "json_schema", json_schema: { name: "action_items", schema: SCHEMA } },
  });
  // CALE-END
  if (aborted()) return;
  let items = [];
  try { items = JSON.parse(reply.choices[0].message.content).items || []; }
  catch { send("actions_error", { message: "The action items were cut short. Try a shorter transcript." }); }
  const seen = new Set();
  items = items.filter((i) => { const k = (i.owner + "|" + i.task).toLowerCase(); return seen.has(k) ? false : (seen.add(k), true); });
  const total = (performance.now() - t0) / 1000;
  send("actions", { items });
  const c2 = reply.cale || {};
  send("done", {
    size: c2.size ?? size ?? meta.size ?? null,
    size2: c2.size ?? null,
    first_s: round(first, 2),
    recap_s: round(recapDone, 1),
    total_s: round(total, 1),
    goal_met: meta.goal_met_actual ?? null,
    goal_met_actions: c2.goal_met_actual ?? null,
    goals_recap: meta.goals_actual ?? null,
    goals_actions: c2.goals_actual ?? null,
    meta_keys: Object.keys(meta), reply_cale_keys: Object.keys(c2),
  });
}

function developerSource() {
  const lines = fs.readFileSync(fileURLToPath(import.meta.url), "utf8").split("\n");
  const out = [];
  let on = false, block = [];
  const flush = () => {
    const ind = Math.min(...block.filter((l) => l.trim()).map((l) => l.match(/^ */)[0].length), 1e9);
    out.push(...block.map((l) => l.slice(ind === 1e9 ? 0 : ind)));
    block = [];
  };
  for (const l of lines) {
    const t = l.trim();
    if (t === "// CALE-BEGIN") on = true;
    else if (t === "// CALE-END") { on = false; flush(); }
    else if (on) block.push(l);
  }
  return out.join("\n");
}

const FILES = {
  "/": ["index.html", "text/html; charset=utf-8"],
  "/index.html": ["index.html", "text/html; charset=utf-8"],
  "/sample": ["../sample/standup.txt", "text/plain; charset=utf-8"],
};

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://127.0.0.1");
  try {
    if (req.method === "GET" && FILES[url.pathname]) {
      const [f, type] = FILES[url.pathname];
      res.writeHead(200, { "Content-Type": type, "Cache-Control": "no-store" });
      return res.end(fs.readFileSync(path.join(HERE, "public", f)));
    }
    if (req.method === "GET" && url.pathname === "/api/source") {
      res.writeHead(200, { "Content-Type": "text/plain; charset=utf-8" });
      return res.end(developerSource());
    }
    if (req.method === "POST" && url.pathname === "/api/run") {
      let body = "";
      for await (const part of req) { body += part; if (body.length > MAX_BODY) { res.writeHead(413).end(); return; } }
      let text = "";
      try { text = String(JSON.parse(body).text || "").trim(); } catch { /* empty */ }
      res.writeHead(200, { "Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-store", Connection: "keep-alive" });
      const send = (event, data) => { if (!res.writableEnded) res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`); };
      if (!text) { send("fail", { message: "Paste a transcript first." }); return res.end(); }
      if (busy) { send("fail", { message: "A recap is already running. Wait for it to finish." }); return res.end(); }
      busy = true;
      let gone = false;
      res.on("close", () => { gone = true; });
      try { await run(text, send, () => gone); }
      catch (e) { console.error("run failed:", e?.message); send("fail", { message: plainError(e) }); }
      finally { busy = false; res.end(); }
      return;
    }
    res.writeHead(404, { "Content-Type": "text/plain" }).end("Not found");
  } catch (e) {
    console.error("request failed:", e?.message);
    if (!res.headersSent) res.writeHead(500, { "Content-Type": "text/plain" });
    res.end("Something went wrong.");
  }
});

let quitting = false;
async function quit(code = 0) {
  if (quitting) return;
  quitting = true;
  try {
    // CALE-BEGIN
    await recapObj?.end(); await actionsObj?.end();       // release the objects (ends the leases)
    // CALE-END
  } catch { /* already gone */ }
  server.close();
  process.exit(code);
}
process.on("SIGINT", () => quit(0));
process.on("SIGTERM", () => quit(0));

server.on("error", (e) => {
  console.error(e.code === "EADDRINUSE" ? `Port ${PORT} is already in use. Is Meeting Recap already running?` : "Server error: " + e.message);
  process.exit(1);
});
server.listen(PORT, "127.0.0.1", () => {
  const addr = `http://127.0.0.1:${PORT}`;
  console.log(`Meeting Recap is running at ${addr}  (Cale: ${CALE_URL})`);
  if (process.env.NO_OPEN !== "1" && !process.env.CALE_NO_BROWSER) {
    const [cmd, args] = process.platform === "darwin" ? ["open", [addr]] : process.platform === "win32" ? ["cmd", ["/c", "start", "", addr]] : ["xdg-open", [addr]];
    try { const c = spawn(cmd, args, { stdio: "ignore", detached: true }); c.on("error", () => {}); c.unref(); } catch { /* fail quietly */ }
  }
});
