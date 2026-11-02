// The slider's math (cale/anythingllm/feel.js). No server, no browser. Run: node cale/tests/chat_feel.js
const feel = require("../anythingllm/feel.js");
const rows = []; const check = (n, ok, d = "") => rows.push([ok ? "PASS" : "FAIL", n, ok ? "" : d]);
const size = (pct, tps, need) => ({ size_pct: pct, decode_tps: tps, needs_gb: need });
// A strong computer: every size fast, memory plenty.
const mac = { ram_gb: 24, sizes: { a: size(100, 35, 4.5), b: size(81, 42, 4.0), c: size(63, 50, 3.6), d: size(48, 62, 3.3) } };
// A weak computer: the largest size is slower than reading speed.
const hp = { ram_gb: 16, sizes: { a: size(100, 5.6, 4.5), b: size(81, 6.8, 4.0), c: size(63, 9.5, 3.6), d: size(48, 13, 3.3) } };
// A computer with little memory: the chat's memory goal (40%) allows only the small sizes.
const tiny = { ram_gb: 8, sizes: { a: size(100, 20, 4.5), b: size(81, 24, 4.0), c: size(63, 28, 3.6), d: size(48, 33, 3.3) } };

const m = feel.ends(mac, 40), h = feel.ends(hp, 40), t = feel.ends(tiny, 40);
check("the ends exist", m && h && t);
check("Mac: the smarter end asks for the largest size's speed", Math.abs(m.lo - 0.9 * 35) < 0.01, m);
check("Mac: the quicker end asks for the fastest size's speed", Math.abs(m.hi - 0.98 * 62) < 0.01, m);
check("HP: the slider sits lower than on the Mac", h.lo < m.lo && h.hi < m.hi, [h, m]);
check("limited memory: the largest size is left out (40% of 8 GB is 3.2 GB: none fit, the smallest is all there is)", t.lo === t.hi || t.hi <= 0.98 * 33 + 1e-9, t);
check("position 1 is the smarter end, 0 the quicker end", feel.goalFor(1, m) === m.lo && feel.goalFor(0, m) === m.hi);
check("the middle is between", feel.goalFor(0.5, m) > m.lo && feel.goalFor(0.5, m) < m.hi);
check("a position outside 0..1 is held to the ends", feel.goalFor(2, m) === m.lo && feel.goalFor(-1, m) === m.hi);
check("Mac default: the smarter end (even the largest size is above reading speed)", feel.defaultPosition(m) === 1);
const hd = feel.defaultPosition(h);
check("HP default: between the ends", hd > 0 && hd < 1, hd);
check("HP default asks for reading speed", Math.abs(feel.goalFor(hd, h) - feel.READING_TPS) < 0.01, feel.goalFor(hd, h));
check("a computer where even the quickest size is below reading speed: the quicker end", feel.defaultPosition({ lo: 2, hi: 4 }) === 0);
check("one allowed size: the slider has no range, default smarter", feel.defaultPosition({ lo: 5, hi: 5 }) === 1);
check("no data yet: not ready", feel.ends(null, 40) === null && feel.ends({ ram_gb: 0, sizes: {} }, 40) === null && feel.ends({ ram_gb: 8 }, 40) === null);
check("sizes without a speed are ignored", feel.ends({ ram_gb: 24, sizes: { a: { size_pct: 100, needs_gb: 4 } } }, 40) === null);
check("no memory goal: every size counts", feel.ends(tiny, undefined).lo === 0.9 * 20);
for (const r of rows) console.log(r[0], "-", r[1], r[2] ? JSON.stringify(r[2]) : "");
process.exit(rows.every((r) => r[0] === "PASS") ? 0 : 1);
