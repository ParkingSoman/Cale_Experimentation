// Regression test for cale/anythingllm/preload.js: a FRESH launch (no cookie yet), and that the code is never the cookie.
// Run: CALE_CHAT_TOKEN=code-ccc CALE_CHAT_COOKIE=cookie-ddd CALE_CHAT_MAIN_PORT=47962 node -r cale/anythingllm/preload.js cale/tests/chat_guard_fresh.js
// Fail-closed check: start it with a missing CALE_CHAT_COOKIE, or the same value for both. It must refuse to start.
// FRESH-LAUNCH CASE: no cookie yet. First load with the code gets the cookie; the code is then spent.
const http = require("http");
const PORT = parseInt(process.env.CALE_CHAT_MAIN_PORT, 10);
const srv = http.createServer((req, res) => { res.end("ok"); });
srv.listen(PORT, "127.0.0.1", async () => {
  const get = (path, headers = {}) => new Promise((resolve) => {
    const r = http.request({ host: "127.0.0.1", port: PORT, path, method: "GET", headers: { Host: `127.0.0.1:${PORT}`, ...headers } }, (res) => { res.resume(); res.on("end", () => resolve({ code: res.statusCode, cookie: res.headers["set-cookie"] })); });
    r.on("error", () => resolve({ code: 0 })); r.end();
  });
  const T = process.env.CALE_CHAT_TOKEN, C = process.env.CALE_CHAT_COOKIE;
  const rows = []; const check = (n, g, w) => rows.push([g === w ? "PASS" : "FAIL", n, g, w]);
  const a = await get(`/?t=${T}`);
  check("first load with the code (no cookie)", a.code, 200);
  check("it sets a cookie that is not the code", !!a.cookie && a.cookie[0].includes(`cale_chat=${C}`) && !a.cookie[0].includes(`cale_chat=${T}`), true);
  check("same code again, no cookie", (await get(`/?t=${T}`)).code, 403);
  check("the code used as the cookie value", (await get("/api/ping", { Cookie: `cale_chat=${T}` })).code, 403);
  check("the real cookie", (await get("/api/ping", { Cookie: `cale_chat=${C}` })).code, 200);
  for (const r of rows) console.log(r[0], "-", r[1], `(got ${r[2]}, want ${r[3]})`);
  process.exit(rows.every((r) => r[0] === "PASS") ? 0 : 1);
});
