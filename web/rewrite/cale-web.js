/* A small browser client for Cale. It mirrors cale/client.py: connect, a model object that lives while a connection stays open, streamed chat.
   No packages. It runs on a web page served over https. The page talks to Cale on the visitor's own computer, at http://127.0.0.1:47821.
   Cale knows the page by its address (the Origin the browser sends), never by a name the page picks. See docs/API.md, "Websites".

     const cale = new CaleWeb();
     if (!(await cale.isRunning())) { ...show "Get Cale"... }
     await cale.connect({ onWait: () => ... });                         // the person sees an Allow question on the computer, once
     const m = await cale.model({ name: "rewriter", min_tokens_per_s: 6, max_tokens: 120 });
     for await (const ev of cale.chat({ model: m.id, messages })) { if (ev.text) ...; if (ev.done) ... }
     m.close();                                                         // or close the tab: the connection ends and so does the object
*/
(function (root) {
  "use strict";

  var DEFAULT_BASE = "http://127.0.0.1:47821";        // 127.0.0.1, not "localhost": Cale refuses any other Host

  /* kind: "missing" (Cale did not answer), "denied", "blocked" (the person said no to this site: Cale will not ask again until they reset it), "wait" (the question went unanswered too often: the site waits 7 days), "busy", "gone" (the key or the object ended), "failed" */
  function CaleWebError(kind, message, status, body) {
    var e = new Error(message);
    e.name = "CaleWebError";
    e.kind = kind; e.status = status; e.body = body || {};
    return e;
  }

  function errorFrom(status, body) {
    var code = body && body.error && body.error.code;
    var msg = (body && body.error && body.error.message) || ("Cale answered " + status);
    var e = CaleWebError("failed", msg, status, body);
    e.code = code;
    if (code === "denied") e.kind = "denied";
    else if (code === "site_blocked") e.kind = "blocked";
    else if (code === "asked_recently") { e.kind = "wait"; e.retryAfterS = body.error.cale && body.error.cale.retry_after_s; }
    else if (code === "busy" || code === "already_waiting") e.kind = "busy";
    else if (status === 401 || code === "wrong_origin" || code === "invalid_api_key") e.kind = "gone";
    return e;
  }

  function CaleWeb(opts) {
    opts = opts || {};
    this.base = (opts.base || DEFAULT_BASE).replace(/\/+$/, "");
    this._fetch = opts.fetch || (root.fetch && root.fetch.bind(root));
    this.storageKey = opts.storageKey || "cale-web-key";
    this._store = opts.storage === undefined ? safeStorage() : opts.storage;
    this.key = this._load();
  }

  function safeStorage() {
    try { return root.localStorage || null; } catch (e) { return null; }       // blocked storage: the page works, it only asks again next time
  }

  CaleWeb.prototype._load = function () {
    try { return (this._store && this._store.getItem(this.storageKey)) || null; } catch (e) { return null; }
  };
  CaleWeb.prototype._save = function (key) {
    this.key = key;
    try { if (this._store) { if (key) this._store.setItem(this.storageKey, key); else this._store.removeItem(this.storageKey); } } catch (e) { /* ignore */ }
  };
  CaleWeb.prototype.forget = function () { this._save(null); };

  /* One request. A network failure means Cale is not there (or the computer said no to the browser): kind "missing". */
  CaleWeb.prototype._req = async function (path, o) {
    o = o || {};
    var headers = {};
    if (o.body !== undefined) headers["Content-Type"] = "application/json";
    if (o.auth !== false && this.key) headers["Authorization"] = "Bearer " + this.key;
    var res;
    try {
      res = await this._fetch(this.base + path, {
        method: o.method || (o.body !== undefined ? "POST" : "GET"), headers: headers, signal: o.signal,
        body: o.body === undefined ? undefined : JSON.stringify(o.body)
      });
    } catch (e) {
      if (e && e.name === "AbortError") throw e;
      throw CaleWebError("missing", "Cale did not answer.");
    }
    return res;
  };

  CaleWeb.prototype._json = async function (path, o) {
    var res = await this._req(path, o);
    var text = await res.text(), body = {};
    try { body = text ? JSON.parse(text) : {}; } catch (e) { /* not JSON */ }
    return { status: res.status, body: body, res: res };
  };

  /* Does Cale answer on this computer? Needs no key and no question. */
  CaleWeb.prototype.isRunning = async function () {
    try {
      var ctl = typeof AbortController !== "undefined" ? new AbortController() : null;
      var timer = ctl && setTimeout(function () { ctl.abort(); }, 3000);
      var r = await this._json("/cale/status", { auth: false, signal: ctl && ctl.signal });
      if (timer) clearTimeout(timer);
      return r.status === 200 && r.body.ok === true;
    } catch (e) { return false; }
  };

  /* Ask the person for a key. Shows an Allow question on the computer, naming this site. Returns the key.
     onWait() is called once if the person has not answered yet. A key saved by an earlier visit is used without a question (see check()). */
  CaleWeb.prototype.connect = async function (o) {
    o = o || {};
    var t0 = Date.now(), maxS = o.maxWaitS || 300, told = false;
    var r = await this._json("/cale/connect", { auth: false, body: { wait_s: 20 } });
    var busyTries = 0;
    while (true) {
      if (r.status === 200 && r.body.api_key) { this._save(r.body.api_key); return r.body.api_key; }
      if (r.status === 202) {
        if (!told) { told = true; if (o.onWait) o.onWait(); }
        if ((Date.now() - t0) / 1000 > maxS) throw CaleWebError("denied", "Nobody answered in time.");
        await sleep(o.pollMs || 1500);
        r = await this._json(r.body.poll, { auth: false });
        continue;
      }
      if (r.status === 409 || (r.status === 429 && r.body.error && r.body.error.code === "busy")) {   // another question is open: wait and ask again
        if (++busyTries > 40 || (Date.now() - t0) / 1000 > maxS) throw errorFrom(r.status, r.body);
        if (!told) { told = true; if (o.onWait) o.onWait(); }
        await sleep(o.pollMs || 1500);
        r = await this._json("/cale/connect", { auth: false, body: { wait_s: 20 } });
        continue;
      }
      throw errorFrom(r.status, r.body);
    }
  };

  /* Is the saved key still good? One cheap call. If not, the key is forgotten. */
  CaleWeb.prototype.check = async function () {
    if (!this.key) return false;
    var r = await this._json("/v1/models");
    if (r.status === 200) return true;
    if (r.status === 401 || r.status === 403) { this.forget(); return false; }
    throw errorFrom(r.status, r.body);
  };

  /* A model object, like cale.client.model: the goals go in, an id comes out. The object lives while this connection stays open.
     Closing it (m.close()), leaving the page, or closing the tab ends it. If Cale restarts, m.alive() becomes false: make a new one. */
  CaleWeb.prototype.model = async function (spec) {
    var ctl = new AbortController();
    var res = await this._req("/cale/models", { body: spec, signal: ctl.signal });
    if (res.status !== 200) {
      var text = await res.text(), body = {};
      try { body = JSON.parse(text); } catch (e) { /* ignore */ }
      throw errorFrom(res.status, body);
    }
    var reader = res.body.getReader(), dec = new TextDecoder(), buf = "", info = null;
    while (info === null) {
      var x = await reader.read();
      if (x.done) throw CaleWebError("failed", "Cale closed the connection.");
      buf += dec.decode(x.value, { stream: true });
      var nl = buf.indexOf("\n");
      if (nl >= 0) { info = JSON.parse(buf.slice(0, nl)); buf = buf.slice(nl + 1); }
    }
    var state = { alive: true };
    (async function drain() {                      // read the heartbeats so they never pile up; the end of the stream means the object is gone
      try { while (true) { var y = await reader.read(); if (y.done) break; } } catch (e) { /* closed */ }
      state.alive = false;
    })();
    return {
      id: info.id, info: info,
      alive: function () { return state.alive; },
      close: function () { state.alive = false; try { ctl.abort(); } catch (e) { /* ignore */ } }
    };
  };

  /* A streamed chat call. An async iterator of {text} pieces, then one {done, size, cale}. Abort with `signal` (Cale then stops the work). */
  CaleWeb.prototype.chat = async function* (body, signal) {
    var res = await this._req("/v1/chat/completions", { body: Object.assign({}, body, { stream: true }), signal: signal });
    if (res.status !== 200) {
      var text = await res.text(), eb = {};
      try { eb = JSON.parse(text); } catch (e) { /* ignore */ }
      var err = errorFrom(res.status, eb);
      if (res.status === 404) err.kind = "gone";
      throw err;
    }
    var size = res.headers.get("X-Cale-Size");
    var reader = res.body.getReader(), dec = new TextDecoder(), buf = "", last = null;
    while (true) {
      var x = await reader.read();
      if (x.done) break;
      buf += dec.decode(x.value, { stream: true });
      var i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        var line = buf.slice(0, i).trim();
        buf = buf.slice(i + 2);
        if (line.indexOf("data:") !== 0) continue;
        var payload = line.slice(5).trim();
        if (payload === "[DONE]") { yield { done: true, size: size, cale: last }; return; }
        var ev = JSON.parse(payload);
        if (ev.error) throw CaleWebError("failed", ev.error.message || "Cale stopped.");
        if (ev.cale) last = ev.cale;
        var piece = ev.choices && ev.choices[0] && ev.choices[0].delta && ev.choices[0].delta.content;
        if (piece) yield { text: piece };
      }
    }
    yield { done: true, size: size, cale: last };
  };

  function sleep(ms) { return new Promise(function (ok) { setTimeout(ok, ms); }); }

  CaleWeb.DEFAULT_BASE = DEFAULT_BASE;
  CaleWeb.Error = CaleWebError;
  if (typeof module !== "undefined" && module.exports) module.exports = CaleWeb;
  else root.CaleWeb = CaleWeb;
})(typeof window !== "undefined" ? window : globalThis);
