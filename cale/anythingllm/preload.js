// Loaded with `node -r preload.js` before AnythingLLM starts (see cale/chat.py). AnythingLLM itself stays unchanged.
// 1. Every server listens on this computer only (AnythingLLM listens on all addresses by default).
// 2. Who may talk to the chat server: only the window program, and nothing from a web page in a browser:
//      a. the Host header must be 127.0.0.1:<port> or localhost:<port>  (stops DNS rebinding);
//      b. an Origin header, if there is one, must be the server's own address (no cross-origin use; no CORS headers are sent by us);
//      c. a per-launch secret: the request must carry the cookie cale_chat=<token>. The window program opens /?t=<token> once;
//         the server then sets that cookie (HttpOnly, SameSite=Strict). Checks run at the http.Server 'request' and 'upgrade'
//         events, before express sees anything.
//    Settings from cale/chat.py: CALE_CHAT_TOKEN, CALE_CHAT_MAIN_PORT (the chat server: all three checks),
//    CALE_CHAT_COLLECTOR_PORT (the document reader: Host and Origin checks only; the chat server calls it).
// 3. Nothing leaves this computer: connections to any address except this computer are refused at the socket layer
//    (CALE_ALLOW_NET=1 lifts this, for tests). The purpose is privacy: what the person types stays here. It is not a wall
//    against an attacker who runs code inside this process. Native add-ons (onnxruntime, lancedb, prisma) open sockets
//    below this layer and are NOT covered; the real protection there is that those parts make no network calls.
"use strict";
const net = require("net");
const http = require("http");
const https = require("https");
const crypto = require("crypto");

// ---- 1. listen on this computer only ----------------------------------------------------------------------------------
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

// ---- 2. who may talk to the server ------------------------------------------------------------------------------------
const TOKEN = process.env.CALE_CHAT_TOKEN || "";
const MAIN_PORT = parseInt(process.env.CALE_CHAT_MAIN_PORT || "0", 10);
const COLLECTOR_PORT = parseInt(process.env.CALE_CHAT_COLLECTOR_PORT || "0", 10);

function same(a, b) {                       // constant-time compare of two strings
  const x = Buffer.from(String(a)), y = Buffer.from(String(b));
  return x.length === y.length && crypto.timingSafeEqual(x, y);
}

function cookieOf(req, name) {
  for (const part of String(req.headers.cookie || "").split(";")) {
    const i = part.indexOf("=");
    if (i > 0 && part.slice(0, i).trim() === name) return part.slice(i + 1).trim();
  }
  return "";
}

/** null when the request may pass; else a short reason. `setCookie` is true when this request carries the right ?t=. */
function judge(req, port, needToken) {
  const host = req.headers.host || "";
  if (host !== `127.0.0.1:${port}` && host !== `localhost:${port}`) return { why: "wrong host" };
  const origin = req.headers.origin;
  if (origin !== undefined && origin !== `http://127.0.0.1:${port}` && origin !== `http://localhost:${port}`) return { why: "wrong origin" };
  if (!needToken) return null;
  if (same(cookieOf(req, "cale_chat"), TOKEN)) return null;
  let t = "";
  try { t = new URL(req.url, "http://x").searchParams.get("t") || ""; } catch { /* no query */ }
  if (t && same(t, TOKEN) && (req.method === "GET" || req.method === "HEAD")) return { setCookie: true };
  return { why: "no key" };
}

const origEmit = http.Server.prototype.emit;
http.Server.prototype.emit = function (event, req, resOrSocket) {
  if ((event === "request" || event === "upgrade") && TOKEN) {
    const addr = this.address && this.address();
    const port = addr && addr.port;
    if (port && (port === MAIN_PORT || port === COLLECTOR_PORT)) {
      const verdict = judge(req, port, port === MAIN_PORT);
      if (verdict && verdict.why) {
        const body = "Not allowed.\n";
        if (event === "request") {
          resOrSocket.writeHead(403, { "Content-Type": "text/plain", "Content-Length": Buffer.byteLength(body), "Connection": "close" });
          resOrSocket.end(body);
        } else {
          resOrSocket.end(`HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\nContent-Length: ${Buffer.byteLength(body)}\r\nConnection: close\r\n\r\n${body}`);
          resOrSocket.destroy();
        }
        return true;
      }
      if (verdict && verdict.setCookie && event === "request") {
        resOrSocket.setHeader("Set-Cookie", `cale_chat=${TOKEN}; HttpOnly; SameSite=Strict; Path=/`);
      }
    }
  }
  return origEmit.apply(this, arguments);
};

// ---- 3. nothing leaves this computer ----------------------------------------------------------------------------------
if (process.env.CALE_ALLOW_NET !== "1") {
  const isLoopback = (h) => {
    if (typeof h !== "string") return false;
    h = h.replace(/^\[|\]$/g, "").toLowerCase();
    if (h === "localhost" || h === "::1" || h === "0:0:0:0:0:0:0:1") return true;
    if (/^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(h)) return true;
    const m = /^::ffff:(127\.\d{1,3}\.\d{1,3}\.\d{1,3})$/.exec(h);
    return !!m;
  };
  let refused = 0;
  const refuse = (where) => {
    refused += 1;
    console.error(`[cale-net] refused a connection to ${where}: Cale keeps everything on this computer (${refused} so far)`);
    return new Error(`Cale keeps everything on this computer: a connection to ${where} was refused.`);
  };

  // The socket layer: http, https/tls and node's fetch (undici) all end up in net.Socket.prototype.connect.
  const origConnect = net.Socket.prototype.connect;
  net.Socket.prototype.connect = function (...args) {
    let opts;
    try { opts = net._normalizeArgs(args)[0]; } catch { opts = {}; }
    if (opts && opts.path) return origConnect.apply(this, args);                  // a unix socket or pipe: on this computer
    const host = (opts && (opts.host || opts.hostname)) || "localhost";           // no host means localhost (Node's own rule)
    if (!isLoopback(host)) {
      const err = refuse(String(host));
      process.nextTick(() => { this.destroy(err); });
      return this;
    }
    return origConnect.apply(this, args);
  };

  // Second layer: a clear early error from the http and fetch calls (what the code asked for, before any lookup).
  const LOCAL = new Set(["127.0.0.1", "localhost", "::1", "[::1]"]);
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
      if (!LOCAL.has(h) && !isLoopback(h)) throw refuse(h);
      return orig.apply(this, args);
    };
  }
  if (typeof globalThis.fetch === "function") {
    const origFetch = globalThis.fetch;
    globalThis.fetch = function (input, init) {
      let h = "";
      try { h = new URL(typeof input === "string" ? input : input.url || String(input)).hostname; } catch { /* not a URL */ }
      if (!LOCAL.has(h) && !isLoopback(h)) return Promise.reject(refuse(h));
      return origFetch.call(this, input, init);
    };
  }
}

// ---- End when the program that started us ends (cale/chat.py sets CALE_PARENT_PID) ------------------------------------
if (process.env.CALE_PARENT_PID) {
  const parent = parseInt(process.env.CALE_PARENT_PID, 10);
  setInterval(() => {
    try { process.kill(parent, 0); } catch (e) { if (e.code === "ESRCH") process.exit(0); }
  }, 2000).unref();
}
