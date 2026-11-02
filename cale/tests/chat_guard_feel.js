// Regression test for the slider route (cale/anythingllm/feel.js, loaded by preload.js): it passes the same guard as every other request.
// Run: CALE_CHAT_TOKEN=code-eee CALE_CHAT_COOKIE=cookie-fff CALE_CHAT_MAIN_PORT=47964 node -r ./cale/anythingllm/preload.js cale/tests/chat_guard_feel.js
// No Cale is running for this test: the route answers "not ready", which is what a computer with no AI loaded yet gets.
const http = require("http");
const PORT = parseInt(process.env.CALE_CHAT_MAIN_PORT, 10);
const srv = http.createServer((req, res) => { res.end("plain server"); });
srv.listen(PORT, "127.0.0.1", async () => {
  const call = (method, path, { headers = {}, body } = {}) => new Promise((resolve) => {
    const r = http.request({ host: "127.0.0.1", port: PORT, path, method, headers: { Host: `127.0.0.1:${PORT}`, ...headers, ...(body ? { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(body) } : {}) } },
      (res) => { let d = ""; res.on("data", (c) => (d += c)); res.on("end", () => resolve({ code: res.statusCode, body: d })); });
    r.on("error", () => resolve({ code: 0, body: "" })); if (body) r.write(body); r.end();
  });
  const C = process.env.CALE_CHAT_COOKIE, ok = { Cookie: `cale_chat=${C}`, Origin: `http://127.0.0.1:${PORT}` };
  const rows = []; const check = (n, g, w) => rows.push([g === w ? "PASS" : "FAIL", n, g, w]);
  check("no cookie: refused", (await call("GET", "/cale-feel")).code, 403);
  check("no cookie, POST: refused", (await call("POST", "/cale-feel", { body: '{"position":0.5}' })).code, 403);
  check("the code as the cookie: refused", (await call("GET", "/cale-feel", { headers: { Cookie: "cale_chat=code-eee" } })).code, 403);
  check("a foreign origin with the cookie: refused", (await call("POST", "/cale-feel", { headers: { Cookie: `cale_chat=${C}`, Origin: "http://evil.example" }, body: '{"position":0.5}' })).code, 403);
  check("a wrong host with the cookie: refused", (await call("GET", "/cale-feel", { headers: { Cookie: `cale_chat=${C}`, Host: "evil.example" } })).code, 403);
  const g = await call("GET", "/cale-feel", { headers: ok });
  check("the cookie: answered", g.code, 200);
  check("no Cale yet: not ready", JSON.parse(g.body).ready, false);
  check("a bad method is refused", (await call("DELETE", "/cale-feel", { headers: ok })).code, 405);
  check("a position above 1 is refused", (await call("POST", "/cale-feel", { headers: ok, body: '{"position":2}' })).code, 400);
  check("a position that is not a number is refused", (await call("POST", "/cale-feel", { headers: ok, body: '{"position":"fast"}' })).code, 400);
  check("a body that is not JSON is refused", (await call("POST", "/cale-feel", { headers: ok, body: "nope" })).code, 400);
  check("a good position with no Cale: not ready (503), nothing breaks", (await call("POST", "/cale-feel", { headers: ok, body: '{"position":0.5}' })).code, 503);
  check("a look-alike path is not taken by the route", (await call("GET", "/cale-feelx", { headers: ok })).body, "plain server");
  check("other paths still reach the server", (await call("GET", "/", { headers: ok })).body, "plain server");
  for (const r of rows) console.log(r[0], "-", r[1], `(got ${r[2]}, want ${r[3]})`);
  process.exit(rows.every((r) => r[0] === "PASS") ? 0 : 1);
});
