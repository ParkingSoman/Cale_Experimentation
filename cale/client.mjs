// A small client for Cale, for Node 18 or newer. No packages. It only wraps the documented HTTP (docs/API.md), so the same calls work from
// any language. Copy this file into your app.
//
//   import { connect, ensureRunning } from "./client.mjs";
//   await ensureRunning();                                         // starts Cale if it is not running (needs the `cale` command)
//   const cale = await connect("My App");                          // asks the person to allow the app, once. Keep cale.key.
//   const model = await cale.createModel({ name: "answers", min_tokens_per_s: 15, memory_pct: 40 });   // a model object on a lease
//   const reply = await cale.chat({ model: model.id, messages: [{ role: "user", content: "Hello" }] });
//   console.log(reply.choices[0].message.content, reply.cale.goals_actual);
//   await model.end();

import { spawnSync } from "node:child_process";

export const DEFAULT_URL = "http://127.0.0.1:47821";

export class CaleError extends Error {
  constructor(status, body) {
    super(`Cale ${status}: ${body?.error?.message ?? JSON.stringify(body)}`);
    this.status = status;
    this.code = body?.error?.code;
    this.body = body;
  }
}

async function request(base, path, { method = "GET", key, body } = {}) {
  const res = await fetch(base + path, {
    method,
    headers: { "Content-Type": "application/json", ...(key ? { Authorization: `Bearer ${key}` } : {}) },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  const json = text ? JSON.parse(text) : {};
  return { status: res.status, headers: res.headers, json };
}

/** Start Cale if it is not running. Needs the `cale` command (installed by cale/install.sh or install.ps1). */
export async function ensureRunning(base = DEFAULT_URL) {
  try { await request(base, "/cale/status"); return true; } catch { /* not running */ }
  const r = spawnSync("cale", ["start", "--wait"], { stdio: "inherit", shell: process.platform === "win32" });
  if (r.status !== 0) throw new Error("Cale did not start. Is it installed? See cale/README.md.");
  return true;
}

/** Ask for a key. The person at the computer allows the app (a dialog, or `cale allow NAME` in a terminal when there is no screen). */
export async function connect(app, { base = DEFAULT_URL, maxWaitS = 300, onWait = (msg) => console.error(msg) } = {}) {
  let r = await request(base, "/cale/connect", { method: "POST", body: { app, wait_s: 20 } });
  const t0 = Date.now();
  let told = false;
  while (r.status === 202) {
    if (!told) { onWait(r.json.message); told = true; }
    if ((Date.now() - t0) / 1000 > maxWaitS) throw new Error("Nobody allowed the app in time.");
    await new Promise((ok) => setTimeout(ok, 2000));
    r = await request(base, r.json.poll);
  }
  if (r.status !== 200) throw new CaleError(r.status, r.json);
  return new Cale(base, r.json.api_key);
}

export class Cale {
  constructor(base, key) {
    this.base = base;
    this.key = key;
  }

  async #call(path, opts = {}) {
    const r = await request(this.base, path, { key: this.key, ...opts });
    if (r.status >= 400) throw new CaleError(r.status, r.json);
    return r;
  }

  /** OpenAI-style chat. Returns the JSON reply (with `cale` fields: goals_actual, size_changes, ...). */
  async chat(body) { return (await this.#call("/v1/chat/completions", { method: "POST", body })).json; }

  /** Streamed chat: an async iterator of the parsed chunks. */
  async *chatStream(body) {
    const res = await fetch(this.base + "/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${this.key}` },
      body: JSON.stringify({ ...body, stream: true }),
    });
    if (res.status >= 400) throw new CaleError(res.status, await res.json());
    let buf = "";
    for await (const part of res.body) {
      buf += Buffer.from(part).toString("utf8");
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, i).trim();
        buf = buf.slice(i + 2);
        if (line.startsWith("data: ") && line !== "data: [DONE]") yield JSON.parse(line.slice(6));
      }
    }
  }

  /** Ask before you run: can this computer meet the goals, and which size would Cale use? Nothing runs. */
  async predict(goals = {}) { return (await this.#call("/cale/predict", { method: "POST", body: goals })).json; }

  async models() { return (await this.#call("/v1/models")).json; }

  /**
   * A model object on a lease: it lives while you use it or renew it, and ends by itself if the app dies.
   * Returns {id, info, renew(), end()}. The lease is renewed in the background every third of its length.
   */
  async createModel({ leaseS = 60, ...spec } = {}) {
    const r = await this.#call("/cale/models", { method: "POST", body: { ...spec, lease_s: leaseS } });
    const id = r.json.id;
    const timer = setInterval(() => this.renewModel(id).catch(() => {}), (leaseS * 1000) / 3);
    timer.unref?.();
    return {
      id, info: r.json,
      renew: () => this.renewModel(id),
      end: async () => { clearInterval(timer); await this.#call("/cale/lease", { method: "POST", body: { id, end: true } }); },
    };
  }

  async renewModel(id) { return (await this.#call("/cale/lease", { method: "POST", body: { id } })).json; }

  /** Keep a size in memory on a lease. Returns {id, info, end()}. */
  async hold({ leaseS = 60, ...spec } = {}) {
    const r = await this.#call("/cale/load", { method: "POST", body: { ...spec, lease_s: leaseS } });
    const id = r.json.hold;
    const timer = setInterval(() => this.#call("/cale/lease", { method: "POST", body: { hold: id } }).catch(() => {}), (leaseS * 1000) / 3);
    timer.unref?.();
    return { id, info: r.json, end: async () => { clearInterval(timer); await this.#call("/cale/lease", { method: "POST", body: { hold: id, end: true } }); } };
  }
}
