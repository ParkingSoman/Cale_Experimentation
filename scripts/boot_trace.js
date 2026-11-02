// Where does a Node server spend its start, in wall time? `NODE_OPTIONS="--require scripts/boot_trace.js"`. Prints (to stderr) the process age when listen() is
// called and when the server is listening, and the age at the first request. Wall time, so waiting for the disk and for virus scanners shows here (a CPU profile does not).
const net = require("net");
const t0 = () => (process.uptime() * 1000).toFixed(0);
console.error(`[boot] preload at ${t0()} ms`);
const orig = net.Server.prototype.listen;
net.Server.prototype.listen = function (...a) {
  console.error(`[boot] listen() called at ${t0()} ms`);
  this.once("listening", () => console.error(`[boot] listening at ${t0()} ms`));
  this.once("connection", () => console.error(`[boot] first connection at ${t0()} ms`));
  return orig.apply(this, a);
};
process.on("exit", () => console.error(`[boot] exit at ${t0()} ms`));
