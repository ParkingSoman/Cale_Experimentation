// Tests the Node client against a running Cale. usage: node scripts/js_client_test.mjs BASE KEY
import { Cale, CaleError } from "../cale/client.mjs";

const [base, key] = [process.argv[2], process.argv[3]];
const cale = new Cale(base, key);
const fails = [];
const check = (name, ok, detail = "") => { console.log((ok ? "PASS " : "FAIL ") + name + (ok ? "" : `  [${detail}]`)); if (!ok) fails.push(name); };

const r = await cale.chat({ model: "gemma-3n:smart", messages: [{ role: "user", content: "Name the capital of France in one word." }], max_tokens: 12 });
check("js: chat", /paris/i.test(r.choices[0].message.content), r.choices[0].message.content);

let text = "", n = 0;
for await (const c of cale.chatStream({ model: "gemma-3n:fast", messages: [{ role: "user", content: "Count to five." }], max_tokens: 30 })) { text += c.choices[0].delta.content ?? ""; n++; }
check("js: streamed chat", text.length > 3 && n > 2, text);

const p = await cale.predict({ min_tokens_per_s: 0.1, prompt_tokens: 100 });
check("js: predict", p.goal_met === true && p.chosen?.size, p);

const m = await cale.createModel({ name: "js-answers", min_tokens_per_s: 0.1, max_tokens: 20, leaseS: 15 });
check("js: model object on a lease", m.id === "js-answers" && m.info.lease.expires_in_s === 15, m.info);
const r2 = await cale.chat({ model: m.id, messages: [{ role: "user", content: "Say hi." }] });
check("js: chat with the object", r2.choices[0].message.content.length > 0 && r2.cale.goals_actual?.speed, r2.cale);
await m.end();
try { await cale.chat({ model: m.id, messages: [{ role: "user", content: "x" }] }); check("js: ended object is gone", false); }
catch (e) { check("js: ended object is gone", e instanceof CaleError && e.status === 404, e.message); }

const h = await cale.hold({ size: "fast", leaseS: 15 });
check("js: hold on a lease", h.id && h.info.size, h.info);
await h.end();

try { await cale.chat({ model: "gemma-3n:auto", messages: [{ role: "user", content: "x" }], cale_deadline_s: 5 }); check("js: removed field is an error", false); }
catch (e) { check("js: removed field is an error", e.code === "unsupported_parameter", e.message); }

console.log(fails.length ? "\nFAILED: " + fails.join(", ") : "\nall passed");
process.exit(fails.length ? 1 : 0);
