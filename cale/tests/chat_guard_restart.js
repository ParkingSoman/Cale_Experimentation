// Regression test for cale/anythingllm/preload.js: the RESTART case (the window already has the cookie when a new one-time code arrives).
// Run: CALE_CHAT_TOKEN=code-aaa CALE_CHAT_COOKIE=cookie-bbb CALE_CHAT_MAIN_PORT=47961 node -r cale/anythingllm/preload.js cale/tests/chat_guard_restart.js
// Tests cale/anythingllm/preload.js directly: a small server on a spare port, with the guard loaded by -r.
const http = require("http");
const PORT = parseInt(process.env.CALE_CHAT_MAIN_PORT, 10);
const srv = http.createServer((req, res) => { res.end("ok"); });
srv.listen(PORT, "127.0.0.1", async () => {
  const get = (path, headers = {}, method = "GET") => new Promise((resolve) => {
    const r = http.request({ host: "127.0.0.1", port: PORT, path, method, headers: { Host: `127.0.0.1:${PORT}`, ...headers } }, (res) => { res.resume(); res.on("end", () => resolve({ code: res.statusCode, cookie: res.headers["set-cookie"] })); });
    r.on("error", () => resolve({ code: 0 })); r.end();
  });
  const T = process.env.CALE_CHAT_TOKEN, C = process.env.CALE_CHAT_COOKIE, ck = { Cookie: `cale_chat=${C}` };
  const rows = [];
  const check = (name, got, want) => { rows.push([name, got, want, got === want ? "PASS" : "FAIL"]); };
  // RESTART CASE: the window already has the cookie; the new code arrives with the cookie.
  check("cookie + new code (a restarted server, window already has the cookie)", (await get(`/?t=${T}`, ck)).code, 200);
  check("then the SAME code, no cookie (must now be spent)", (await get(`/?t=${T}`)).code, 403);
  check("cookie only still works", (await get("/api/ping", ck)).code, 200);
  check("wrong code, no cookie", (await get("/?t=nope")).code, 403);
  for (const r of rows) console.log(r[3], "-", r[0], `(got ${r[1]}, want ${r[2]})`);
  process.exit(rows.every((r) => r[3] === "PASS") ? 0 : 1);
});
