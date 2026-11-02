// Profile the start of a Node program: `NODE_OPTIONS="--require scripts/node_profile_preload.js" PROF_OUT=/some/prefix PROF_SECONDS=6 ...`
// After PROF_SECONDS it writes PROF_OUT.<pid>.cpuprofile (open it in Chrome's devtools, or read it with scripts/node_profile_report.py).
const inspector = require("inspector"), fs = require("fs");
const s = new inspector.Session();
s.connect();
s.post("Profiler.enable");
s.post("Profiler.setSamplingInterval", { interval: 500 });
s.post("Profiler.start");
setTimeout(() => s.post("Profiler.stop", (e, r) => { if (!e) fs.writeFileSync(`${process.env.PROF_OUT}.${process.pid}.cpuprofile`, JSON.stringify(r.profile)); }), Number(process.env.PROF_SECONDS || 6) * 1000).unref();
