---
name: feedback-testing-rules
description: "Testing rules that protect Shonu's computer: Claude in Chrome only, never hit the real Cale (port 47821), hidden-zoo anti-overfit review (merged)"
metadata:
  type: feedback
---

## Use Claude in Chrome only (2026-10-14)
- Quote: "from now on you should use claude in chrome since other ways take control of the computer."
- Why: Playwright windows, desktop scripting (osascript / System Events clicks) and computer-use tools grab mouse and keyboard; he cannot use his Mac meanwhile.
- How to apply: check web pages (Cale Hub, demo apps, Open WebUI) with `mcp__claude-in-chrome__*`. Load tools with one ToolSearch call. Start with `tabs_context_mcp`, make my own tab, close it after. Do not use Playwright MCP, chrome-devtools MCP, cua-driver, or osascript/System Events to click dialogs. Do not click the native Mac "Allow" dialog by script: use `CALE_AUTO_APPROVE` in tests or ask Shonu. Subagents for browser checks must get this rule in their prompt (agents never read memory). The HP rule about clicking dialogs is for the HP only ([[reference-hp-testbench]]).

## Never reach the real Cale (2026-10-17, 10-18)
- Shonu said twice that Allow dialogs on his screen are annoying and he will not press them (one was "Allow E2E Test to use Cale" from `cale/tests/e2e.py`).
- Why: `e2e.py` defaulted to `--base http://127.0.0.1:47821`, his REAL Cale. `CALE_DEBUG=1 CALE_AUTO_APPROVE=1` in my shell does nothing for an already running Cale. Each run showed a dialog and left a test key in `keys.json`.
- How to apply: start a TEST Cale on a spare port (dev server 47899) with a fresh `CALE_HOME` and both switches in ITS environment; pass `--base` (now required; 47821 refused). `cale/tests/web_origin.py` and `oslayer.py` are isolated and safe. Never `pkill -f cale.server` or `pkill -f "cale.chat"` (can hit the real Cale/chat): stop test programs by exact path or port. Never UI-script dialogs. The switches are for test copies only. Second incident 2026-10-18: `pressure.py` had no `--base` (default 47821) and asked the real Cale for access; fixed (requires `--base`). Before running older tests in a loop, grep for `47821` or pass `--base http://127.0.0.1:47899`.

## Hidden app zoo review against overfitting (2026-10-12)
- Use a review agent while building so I do not overfit to the demo apps. After the phase he will test me by having me build diverse apps using only the runtime.
- How to apply: reviewer agent gets ONLY the public docs and `docs/PRINCIPLES.md`; writes probe apps in a folder outside the repo (agent with tools, JSON extraction, long-context Q&A, two apps at once, Node.js app, `curl` script, battery job, low-memory setting, streaming backend); returns a scorecard (app types x environments), reporting what the app needed, not code fixes. A second reviewer reads each stage's diff: does this fit the demo apps or one machine? Put rules in the repo ([[lesson-rules]] L124: agents never read memory). See [[project-cale-vision]].

Related: [[reference-github-vercel]].
