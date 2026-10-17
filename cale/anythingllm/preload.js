// Loaded with `node -r preload.js` before AnythingLLM starts (see cale/chat.py).
// 1. Every server listens on this computer only (AnythingLLM listens on all addresses by default).
// 2. Nothing leaves this computer: calls to any other address are refused (CALE_ALLOW_NET=1 lifts this, for tests).
// AnythingLLM itself stays unchanged.
"use strict";
const net = require("net");
const http = require("http");
const https = require("https");

const origListen = net.Server.prototype.listen;
net.Server.prototype.listen = function (...a) {
  if (a[0] && typeof a[0] === "object") {
    if (!a[0].path) a[0] = { ...a[0], host: "127.0.0.1" };
  } else if (a[0] !== undefined && typeof a[0] !== "string") {
    if (typeof a[1] !== "string") a.splice(1, 0, "127.0.0.1");
  } else if (typeof a[0] === "string" && /^\d+$/.test(a[0])) {
    if (typeof a[1] !== "string") a.splice(1, 0, "127.0.0.1");
  }
  return origListen.apply(this, a);
};

if (process.env.CALE_ALLOW_NET !== "1") {
  const LOCAL = new Set(["127.0.0.1", "localhost", "::1", "[::1]", "0.0.0.0"]);
  const refuse = (what) => new Error(`Cale keeps everything on this computer: a call to ${what} was refused.`);
  const hostOf = (args) => {
    const a = args[0];
    if (typeof a === "string") { try { return new URL(a).hostname; } catch { return ""; } }
    if (a instanceof URL) return a.hostname;
    const o = a || {};
    return o.hostname || (o.host || "").replace(/:\d+$/, "") || "localhost";
  };
  for (const [mod, name] of [[http, "request"], [http, "get"], [https, "request"], [https, "get"]]) {
    const orig = mod[name];
    mod[name] = function (...args) {
      const h = hostOf(args);
      if (!LOCAL.has(h)) throw refuse(h);
      return orig.apply(this, args);
    };
  }
  if (typeof globalThis.fetch === "function") {
    const origFetch = globalThis.fetch;
    globalThis.fetch = function (input, init) {
      let h = "";
      try { h = new URL(typeof input === "string" ? input : input.url || String(input)).hostname; } catch {}
      if (!LOCAL.has(h)) return Promise.reject(refuse(h));
      return origFetch.call(this, input, init);
    };
  }
}

// 3. End when the program that started us ends (cale/chat.py sets CALE_PARENT_PID).
if (process.env.CALE_PARENT_PID) {
  const parent = parseInt(process.env.CALE_PARENT_PID, 10);
  setInterval(() => {
    try { process.kill(parent, 0); } catch (e) { if (e.code === "ESRCH") process.exit(0); }
  }, 2000).unref();
}
