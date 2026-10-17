// Checks the logic of web/rewrite/cale-web.js with Node (no browser). usage: node cale/tests/web_client.mjs [--real http://127.0.0.1:PORT]
//   Part 1 talks to a fake Cale made here (stream parsing, polling, errors).
//   Part 2 (only with --real) talks to a real Cale server started with CALE_WEB_DEV_ORIGINS=http://127.0.0.1:5555 CALE_DEBUG=1 CALE_AUTO_APPROVE=1.
//   Node's fetch sends no Origin, so part 2 adds one by hand, as a browser would.
import http from "node:http";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";

const require = createRequire(import.meta.url);
const CaleWeb = require(path.join(path.dirname(fileURLToPath(import.meta.url)), "../../web/rewrite/cale-web.js"));
const fails = [];
const check = (name, ok, detail = "") => { console.log((ok ? "PASS " : "FAIL ") + name + (!ok && detail ? `  [${detail}]` : "")); if (!ok) fails.push(name); };
const memory = () => { const m = new Map(); return { getItem: (k) => m.get(k) ?? null, setItem: (k, v) => m.set(k, v), removeItem: (k) => m.delete(k) }; };

// ---- Part 1: a fake Cale ----------------------------------------------------------------------------------------------
let polls = 0, seen = [];
const fake = http.createServer((req, res) => {
  let body = ""; req.on("data", (d) => body += d); req.on("end", () => {
    seen.push({ method: req.method, url: req.url, auth: req.headers.authorization, ctype: req.headers["content-type"], body });
    const json = (code, o) => { res.writeHead(code, { "Content-Type": "application/json" }); res.end(JSON.stringify(o)); };
    if (req.url === "/cale/status") return json(200, { ok: true });
    if (req.url === "/cale/connect" && req.method === "POST") {
      const b = JSON.parse(body || "{}");
      if (b.mode === "x") return json(200, {});
      if (fake.mode === "denied") return json(403, { error: { code: "denied", message: "no" } });
      if (fake.mode === "cooldown") return json(429, { error: { code: "asked_recently", message: "wait", cale: { retry_after_s: 599 } } });
      if (fake.mode === "poll") return json(202, { status: "pending", ticket: "t-1", poll: "/cale/connect/status?ticket=t-1" });
      return json(200, { app: "site", api_key: "cale-KEY", origin: "https://x.example" });
    }
    if (req.url.startsWith("/cale/connect/status")) return ++polls < 3 ? json(202, { poll: req.url }) : json(200, { api_key: "cale-POLLED" });
    if (req.url === "/v1/models") return req.headers.authorization === "Bearer cale-GOOD" ? json(200, { data: [] }) : json(403, { error: { code: "wrong_origin", message: "x" } });
    if (req.url === "/cale/models") {
      res.writeHead(200, { "Content-Type": "application/x-ndjson" });
      res.write(JSON.stringify({ id: "rewriter", object: "cale.model" }) + "\n");
      const t = setInterval(() => res.write("\n"), 50);
      req.on("close", () => { clearInterval(t); fake.modelClosed = true; });
      return;
    }
    if (req.url === "/v1/chat/completions") {
      if (fake.mode === "nomodel") return json(404, { error: { code: "model_not_found", message: "gone" } });
      res.writeHead(200, { "Content-Type": "text/event-stream", "X-Cale-Size": "62" });
      if (fake.mode === "slow") { res.write('data: {"choices":[{"delta":{"content":"A"}}]}\n\n'); req.on("close", () => { fake.slowClosed = true; }); return; }
      const ev = (o) => "data: " + JSON.stringify(o) + "\n\n";
      const out = ev({ choices: [{ delta: { role: "assistant", content: "" } }] }) + ev({ choices: [{ delta: { content: "Hel" } }] }) +
                  ev({ choices: [{ delta: { content: "lo" } }] }) + ev({ choices: [{ delta: {}, finish_reason: "stop" }], cale: { goal_met_actual: true } }) + "data: [DONE]\n\n";
      for (let i = 0; i < out.length; i += 7) res.write(out.slice(i, i + 7));      // pieces cut in the middle of events
      return res.end();
    }
    json(404, {});
  });
});
await new Promise((ok) => fake.listen(0, "127.0.0.1", ok));
const base = `http://127.0.0.1:${fake.address().port}`;

let c = new CaleWeb({ base, storage: memory() });
check("isRunning true", await c.isRunning());
check("isRunning false when nothing answers", !(await new CaleWeb({ base: "http://127.0.0.1:1", storage: memory() }).isRunning()));
let err = await new CaleWeb({ base: "http://127.0.0.1:1", storage: memory() }).connect().catch((e) => e);
check("connect with no Cale -> kind missing", err.kind === "missing", err.kind);

fake.mode = "ok";
check("connect returns and saves the key", (await c.connect()) === "cale-KEY" && c.key === "cale-KEY");
check("connect sent no app name and no key", JSON.parse(seen.find((s) => s.url === "/cale/connect").body).app === undefined && seen.find((s) => s.url === "/cale/connect").auth === undefined);
const store = memory(); store.setItem("cale-web-key", "cale-SAVED");
check("a saved key is read at start", new CaleWeb({ base, storage: store }).key === "cale-SAVED");
check("works with no storage at all", new CaleWeb({ base, storage: null }).key === null);
const broken = { getItem() { throw new Error("blocked"); }, setItem() { throw new Error("blocked"); }, removeItem() { throw new Error("blocked"); } };
const cb = new CaleWeb({ base, storage: broken });
check("blocked storage does not break connect", (await cb.connect()) === "cale-KEY");

fake.mode = "poll"; polls = 0; let waited = 0;
const c2 = new CaleWeb({ base, storage: memory() });
check("connect polls until the person answers", (await c2.connect({ onWait: () => waited++, pollMs: 10 })) === "cale-POLLED" && waited === 1);
fake.mode = "denied";
err = await new CaleWeb({ base, storage: memory() }).connect().catch((e) => e);
check("denied -> kind denied", err.kind === "denied", err.kind);
fake.mode = "cooldown";
err = await new CaleWeb({ base, storage: memory() }).connect().catch((e) => e);
check("cooldown -> kind wait with retryAfterS", err.kind === "wait" && err.retryAfterS === 599, JSON.stringify([err.kind, err.retryAfterS]));

const c3 = new CaleWeb({ base, storage: memory() }); c3._save("cale-BAD");
check("check() false and key forgotten on 403", (await c3.check()) === false && c3.key === null);
c3._save("cale-GOOD");
check("check() true", (await c3.check()) === true);

const m = await c3.model({ name: "rewriter", min_tokens_per_s: 6, max_tokens: 120 });
const mreq = seen.filter((s) => s.url === "/cale/models").pop();
check("model sends goals with the key", mreq.auth === "Bearer cale-GOOD" && JSON.parse(mreq.body).min_tokens_per_s === 6);
check("model id comes from the first line", m.id === "rewriter" && m.alive());
await new Promise((ok) => setTimeout(ok, 200));
check("model stays alive through heartbeats", m.alive());
m.close();
await new Promise((ok) => setTimeout(ok, 200));
check("closing the model closes the connection", fake.modelClosed === true && !m.alive());

fake.mode = "ok";
let text = "", done = null;
for await (const ev of c3.chat({ model: "rewriter", messages: [{ role: "user", content: "x" }] })) { if (ev.text) text += ev.text; if (ev.done) done = ev; }
check("chat streams pieces cut mid-event", text === "Hello", text);
check("chat ends with size and cale info", done && done.size === "62" && done.cale.goal_met_actual === true);
const creq = seen.filter((s) => s.url === "/v1/chat/completions").pop();
check("chat asks for a stream", JSON.parse(creq.body).stream === true);
fake.mode = "nomodel";
err = await (async () => { try { for await (const _ of c3.chat({ model: "x", messages: [] })) { /* */ } } catch (e) { return e; } })();
check("chat on a gone object -> kind gone", err && err.kind === "gone", err && err.kind);
fake.mode = "slow";
const ctl = new AbortController();
err = await (async () => { try { for await (const ev of c3.chat({ model: "x", messages: [] }, ctl.signal)) { ctl.abort(); } } catch (e) { return e; } })();
await new Promise((ok) => setTimeout(ok, 200));
check("abort stops the stream and closes the connection", err && err.name === "AbortError" && fake.slowClosed === true, err && err.name);
fake.close();

// ---- Part 2: the real server ------------------------------------------------------------------------------------------
const ri = process.argv.indexOf("--real");
if (ri > 0) {
  const real = process.argv[ri + 1];
  const ORIGIN = "http://127.0.0.1:5555";
  const withOrigin = (origin) => (url, o = {}) => fetch(url, { ...o, headers: { ...(o.headers || {}), Origin: origin } });
  const w = new CaleWeb({ base: real, storage: memory(), fetch: withOrigin(ORIGIN) });
  check("real: status from an allowed origin", await w.isRunning());
  const key = await w.connect();
  check("real: connect gives a key", key.startsWith("cale-"));
  check("real: check() works with the key", await w.check());
  const o = await w.model({ name: "rewriter", min_tokens_per_s: 6, max_tokens: 120 }).catch((e) => e);
  // With no model on the computer the server cannot predict speed (an engine problem, not a web problem), so this one is checked only if it can run.
  if (o && o.id) check("real: model object created (goals)", o.id === "rewriter");
  else console.log("SKIP real: model object (this server has no model): " + (o && o.message));
  if (o && o.close) o.close();
  const other = new CaleWeb({ base: real, storage: memory(), fetch: withOrigin("http://127.0.0.1:5556") });
  check("real: an origin that is not allowed sees no Cale", !(await other.isRunning()));
  const stolen = new CaleWeb({ base: real, storage: memory(), fetch: fetch }); stolen._save(key);
  check("real: the site key is refused without its origin", (await stolen.check()) === false);
}
console.log(fails.length ? "\nFAILED: " + fails.join(", ") : "\nAll checks passed.");
process.exit(fails.length ? 1 : 0);
