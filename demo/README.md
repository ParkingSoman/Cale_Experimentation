# The demo (2026-10-17): read this section first

Cale is a real program now. It has its own windows and its own chat. Nothing on screen shows a web address, a console, or a technical word.
The look is a cartoon carrot in sunglasses (`cale/brand/README.md`). The presenter types nothing.

## The story, in three parts

| Part | What the audience sees | What it proves |
|---|---|---|
| 1. Cale | Double-click Cale. A carrot grows out of the soil. "You are all set". Click "Start chatting". Ask a question. Add a document and ask about it. | One download gives a person an AI chat. No sign-up, no usage limit. |
| 2. A desktop app | Double-click Quick Notes. It opens in its own window. Grey ghost text appears while you type. | A downloaded app uses the same AI on this computer. The app is tiny (145 KB on the Mac, 39 KB on Windows). |
| 3. A website | Open the Rewrite website in Chrome. Pick a tone. New text streams in. Cale asks once: "Allow the website ... to use Cale?" | A real website uses the same AI on this computer. |

Say out loud: "Same AI, on this computer, for a chat, for a desktop app, and for a website. Nobody signed up. Nothing has a usage limit. It feels the same on a new Mac and on a ten-year-old laptop."

## Start state (already done before the demo)

| Computer | Files already downloaded |
|---|---|
| Mac | `dist/Cale-mac.dmg` (563 MB), `dist/apps/Quick-Notes-mac.dmg` (145 KB) |
| HP Windows | `Cale-Setup.exe` (342 MB), `Quick-Notes-Setup.exe` (39 KB). Built by `build_exe.ps1` and `build_app_exe.ps1` on the HP. |
| Website | Not published yet. See "Open items". |

The install itself is a recorded time-lapse. Do not run it live.

| Computer | Where the installers are | Check |
|---|---|---|
| Mac | `~/Downloads/Cale-mac.dmg`, `~/Downloads/Quick-Notes-mac.dmg` | Same bytes as `dist/`. No quarantine mark, so no warning. |
| HP | `C:\Users\Shonu\Downloads\Cale-Setup.exe` (359,008,408 B, SHA-256 `1769E88E...F3FD3C`), `Quick-Notes-Setup.exe` (41,616 B, SHA-256 `BE27B6AD...E3400E`) | Signed with the demo certificate (Valid). The copies were not run from Downloads. |

## Speed bar (decided 2026-10-18): the same feel on every computer

A consumer cannot warm up first, so the demo does not either. If a moment is slow on the HP, that is a gap to fix, not a step to hide.
The bar is sameness: each moment must feel the same on the Mac and on the HP. Only the intelligence level may differ. Plan: `~/.claude/plans/okay-overall-i-think-streamed-sutton.md`.

| State | What |
|---|---|
| Done, Mac only | The chat window no longer waits for the AI to load. Mac: chat ready at 0.9 s, was about 5 s. A message sent during the load waits and is answered (`cale/tests/pending_object.py`, and a real run through the chat). |
| Not yet checked on the HP | The same change on Windows. The HP numbers below (40 s and 75 s) are from BEFORE this change. Measure again with `python scripts/timeline_report.py`. |
| Not done yet | Windows starts its browser part before it starts Cale (`CaleWindow.cs`). The marks for the window program itself. The slider. The faster install. |
| Still true | On the HP, close other programs first. McAfee and Defender scans made the install take 23 to 25 minutes (11 on a quiet day). |

Timeline: every open writes marks to `timeline.jsonl` in the Cale folder. `python scripts/timeline_report.py [FILE ...]` prints seconds per step. Give one file per computer to see them side by side.
Chat check (the chat part of the scorecard): `python scripts/chat_race.py NEWFOLDER --bundle dist/chat --node NODE [--cold] [--feel]`. It starts its own test Cale and chat, sends one message at once, prints every step, and with `--feel` moves the Quicker/Smarter slider and shows the size Cale used. Measured 2026-10-18: Mac chat ready 0.9 s, HP 5.7 to 6.2 s (10 to 14 s on the first run after a pause). Details: `docs/HANDOFF-2026-10-18.md`.
The installers in Downloads and on the HP are from BEFORE this work (old chat, no slider). Rebuild them before a demo.

## Measured on 2026-10-17 (fresh test install, real AI, nothing of the owner's real Cale touched)

| Step | Mac (M5) | HP Windows |
|---|---|---|
| Install, start to "You are all set" | 150 to 161 s | 25 min on a heavy-load day (11 min on a normal one) |
| Chat opens by itself after install | within 12 s | 3 s (page ready after 40 s on a warm start) |
| First real chat answer | 1.85 s in total | first text 1.35 s, 5.3 s in total |
| Quick Notes first word (goal 1 s) | 0.12 to 0.15 s | 0.13 to 0.14 s. The first call after a cold start took 1.24 s (goal missed). |

## The website step: two one-time questions

On a Chrome that has never seen the site, the first visit shows two questions. Answer both once and the audience never sees them again.

| Order | Who asks | What it says | What to do |
|---|---|---|---|
| 1 | Chrome (near the address bar) | `cale-rewrite.vercel.app wants to: Access other apps and services on this device` with Block and Allow | Choose **Allow**. The page shows "One quick question from your browser" while it waits. It waits patiently (up to 3 minutes) and never cancels the question. |
| 2 | Cale (a window on the computer) | `Allow the website "cale-rewrite.vercel.app" to use Cale on this computer?` | Choose **Allow**. |

If you choose Block in Chrome, the page says "Your browser is blocking this site", and Chrome's site settings must allow it again. If a question is ignored 4 times, Chrome blocks the site for 7 days, so do not let it time out.

Test hook (for tests only): add `?cale=http://127.0.0.1:<port>` to the address to point the page at a TEST Cale. Only a loopback address with a port is accepted. A yellow "Test mode" banner shows, and the saved key has its own name per address, so the real Cale's key is never used for a test address.

## Safe testing (read before running any test)

| Rule | Why |
|---|---|
| Never run a test against the real Cale (port 47821). | A real Cale shows Allow questions on the screen and keeps test keys. `cale/tests/e2e.py` now refuses that port. |
| Start a TEST Cale: fresh `CALE_HOME`, spare `CALE_PORT`, and `CALE_DEBUG=1` with `CALE_AUTO_APPROVE=1` in its own environment. | These two switches skip the Allow question. They are for test copies only. |
| Isolated tests (safe): `cale/tests/web_origin.py`, `cale/tests/oslayer.py`, `cale/tests/chat_guard_*.js`. | They use a stub engine. |
| On the HP set `CALE_NO_PATH=1` for tests. | It keeps the test from changing the user's PATH. |

## What protects the demo (security, in plain words)

| Piece | Rule |
|---|---|
| Chat | A random one-time code in the window's address, then a cookie. Wrong Host or foreign Origin: refused. The server refuses to start if its two secrets are missing or equal. |
| Websites | Only secure (`https`) sites may ask. Each site gets one Allow question. Each site is counted on its own, with Chromium's numbers: 3 "Don't Allow" presses or 4 unanswered questions mean 7 days of waiting. `cale sites` lists them and `cale forget SITE` resets one. |
| Outside network | The chat may talk to loopback addresses only. |

## Build

| Want | Commands |
|---|---|
| Mac | `python3 scripts/build_chat.py ... --out dist/chat` once, then `python3 scripts/build_installer.py mac` and `python3 scripts/build_app.py mac quick-notes` |
| Windows | `python3 scripts/build_chat.py ... --target win-x64 --out dist/chat-win`, then `python3 scripts/build_installer.py win` and `python3 scripts/build_app.py win quick-notes`. Copy the two `dist` folders to the HP (let the HP download them: scp fails near 90 MB). Run `build_exe.ps1` and `build_app_exe.ps1` there. |

## Open items

| Item | State |
|---|---|
| Rewrite website | LIVE and tested in real Chrome 155: https://cale-rewrite.vercel.app (Vercel project `cale-rewrite`, team "Shonu S's projects"). Files in `web/rewrite/`. A real rewrite streamed back from a test Cale on this Mac. Chrome's local-network permission for this site is already "Allowed" in the owner's Chrome. Not tested: Safari and Firefox, and a Chrome that has never seen the site (it needs the two one-time questions below). |
| Chat, light theme | Being fixed. The dark theme looks right. |
| Linux | No chat, no window program. Covered by the scorecard and by speaking. |
| A first install on a Mac with no developer tools | Not tested. |
| The real taskbar name on Windows | Only the window title and icon were seen. |
| Quick Notes ghost text sometimes repeats part of the prompt | Not investigated. |
| Document reader port in the chat | Loopback only, Host and Origin checked, no cookie. A small accepted risk. |

# Older notes (2026-10-16): the browser-based demo

Kept for the funding pitch and the measured numbers. The screens described in this part (browser pages, the Cale Hub) are no longer the demo.


# The Cale demo: download once, every app works

One download. No terminal. No model choice. After that, five apps from five "developers" just work.
The same five apps run on a fast Mac and on an old laptop. The apps do not change. Cale picks the size.

This file is the run sheet. The numbers to quote are in `results/demo/` (measured, one JSON per computer).

## What the audience sees

| Step | What happens | What to say |
|---|---|---|
| 1. Download | One file per OS. Mac: `Cale.app` (in a `.dmg`). Windows: `Cale-Setup.exe`. Linux: `Cale-linux-x86_64.run`. | "That is the whole install. No Python, no model picker, no settings." |
| 2. Open it | A page opens: the Cale Hub. One progress bar: download, "Fitting Cale to this computer", done. | "Cale measures this computer once. The apps will never ask." |
| 3. Open an app | The Hub lists five apps. Click Start. A native question asks: allow this app to use Cale? Click Allow. | "Each app asks once, like a camera permission." |
| 4. Use all five | The Hub shows live meters per app: size, tokens/s, first word, goal met. A "Computer feels" meter shows if the screen stays smooth. | "The apps asked for a feel, not a model. Watch the sizes change on the old laptop. Watch the feel stay the same." |
| 5. Switch computer | Same five apps on the old HP laptop. | "Smaller sizes. Same feel. The one exception is the app that chose quality over speed, and it says so up front." |

## The centerpiece: absolute against relative (Quick Notes)

The demo story has two claims. (1) One install and the same app code on every computer. (2) The developer controls how the app feels, on any hardware, by stating goals.
Claim 2 is shown with one app and one test. The same notes-style completion call is written three ways, and each is scored against the same target (first word within 1 s, at least 8 tok/s):

| Way | The model line |
|---|---|
| Absolute, full model | `model = "gemma-3n:smart"` |
| Absolute, small model | `model = "gemma-3n:fast"` |
| Relative, goals | `max_first_token_s = 1`, `min_tokens_per_s = 8` |

Run on each computer, against its running Cale: `python3 scripts/ab_absolute_relative.py --base URL --label NAME --out results/demo/ab-NAME.json`

| Computer | Absolute, full | Absolute, small | Relative (model size that ran) |
|---|---|---|---|
| Mac (M5) | 18 of 18 | 18 of 18 (but only 48% of the model) | 18 of 18 (100%) |
| Old HP, Windows | 0 of 18 | 15 of 18 | 14 of 18 (48%) |
| Old HP, Linux | 0 of 18 | 13 of 18 | 14 of 18 (48%) |

Combined score (`python3 scripts/ab_score.py`, saved in `results/demo/ab-combined.json`): a call off target scores 0, a call on target scores the quality of the slice that ran (1.00 full, 0.81 smallest; share of the full model's next-word choices it repeats).

| Score | Mac | HP Windows | HP Linux | Worst | Mean |
|---|---|---|---|---|---|
| Absolute, full | 1.00 | 0.00 | 0.00 | 0.00 | 0.33 |
| Absolute, small | 0.81 | 0.67 | 0.58 | 0.58 | 0.69 |
| Relative, goals | 1.00 | 0.63 | 0.63 | 0.63 | 0.75 |

Goals match the winner on each computer without being told which computer it is. On the laptop they tie the small model (within noise). Limit: quality is agreement with the full model, not a rating of the notes.

Say: "Write the full model and it fails on the old laptop. Write the small model and the Mac loses half its intelligence for nothing. Write the goal, and each computer gets the biggest slice that keeps the feel."
Say before asked: the two HP rows are one laptop. The relative arm misses 4 of 18 on the laptop because 8 tok/s is close to its ceiling.
The other four apps are not the demo. They show where the size ladder (100% down to 48%) is too narrow: that is the funding case for a model with a much wider range of slices.

## The five apps (what each developer values)

| App | Port | Language | The developer asked Cale for | What it adds to the Cale equation |
|---|---|---|---|---|
| Chat (Open WebUI, unmodified) | 8810 | Python | a minimum speed (`min_tokens_per_s`) and a memory share | The most common AI app works everywhere and keeps its speed. |
| Quick Notes | 8821 | Python | a first word in about a second (`max_first_token_s`) | An instant-feel app type that a cloud call cannot match. |
| Private Doc Reader | 8822 | Python | always the full size (`gemma-3n:smart`), told the time first (`/cale/predict`) | Same quality on every computer. Private documents stay on the computer. |
| Background Sorter | 8823 | Python | low power and low memory (`power=light`, `memory_pct`, JSON schema) | Works in the background without slowing the foreground app. |
| Meeting Recap | 8824 | Node.js | a speed goal through the Node client | Any language works. No Python in the app. |

Each app has a "Developer view" switch. It shows the few lines of Cale code the app uses.
All four new apps were written from `docs/API.md` and the public client files only. Their friction notes fixed the docs.

## Build the downloads (on a Mac)

| Want | Command | Result |
|---|---|---|
| Mac | `python3 scripts/build_installer.py mac` | `dist/Cale-mac.dmg`, `dist/Cale-mac.zip` |
| Linux | `python3 scripts/build_installer.py linux` | `dist/Cale-linux-x86_64.run` (`--arch arm64` for ARM) |
| Windows | `python3 scripts/build_installer.py win`, then on Windows run `build_exe.ps1` in `dist/win` | `Cale-Setup.exe` |

Each download carries its own `uv` and Node. It does not carry the AI model (4.4 to 5.5 GB). The installer fetches it and shows progress.

## No "unknown developer" warning (demo computers only)

A public release needs an Apple Developer ID ($99 a year) and a Windows signing certificate. For the two demo computers, do this once:

| Computer | What to do | Undo |
|---|---|---|
| Mac | After the browser downloads `Cale-mac.dmg`, run `xattr -d com.apple.quarantine ~/Downloads/Cale-mac.dmg` before you open it. Do this before the demo, not on stage. | Nothing to undo. It changes one file. |
| Windows (as administrator) | `powershell -ExecutionPolicy Bypass -File prep_windows_demo.ps1`, then build the `.exe`. | same command with `-Undo` |

Why the Mac line works: a browser marks every download with a "quarantine" label. Gatekeeper checks only marked files. Tested on 2026-10-15: Chrome marked the file, Gatekeeper said `rejected`, and the app inside the disk image had no mark after the label was removed.
Do not use `scripts/prep_mac_demo.sh`. On macOS 26 it needs a manual confirm in System Settings and it lowers protection for every app.

## Record the install

Record one real install per computer and speed it up. The live demo starts with Cale installed.
Set `CALE_HOME` to a new empty folder to repeat an install without touching a real one. Test switches (never for real use):

| Variable | Meaning |
|---|---|
| `CALE_HOME` | install folder |
| `CALE_PORT`, `CALE_HUB_PORT` | Cale port (default 47821), Hub port (default 47800) |
| `CALE_NO_LAUNCH_AGENT=1` | Mac: do not touch the real LaunchAgent |
| `CALE_AUTO_APPROVE=1` with `CALE_DEBUG=1` | skip the Allow question (tests only) |
| `CALE_MODEL_SRC` | link a local model instead of downloading |

## The scorecard

Run on each computer, with its Hub:

```
python demo/check_apps.py --hub http://127.0.0.1:47800 --label mac --out results/demo/mac.json
```

It starts each app, does the app's main job through the app's own web calls, and reads the numbers from the Hub.
Verdicts: WELL (done right, every goal Cale reported was met), POOR (done, but a goal was missed or it was very slow), FAIL.
Results: see `results/demo/` (to be filled in by the runs on the Mac, HP Windows and HP Linux).

## Who goes first (the part the developer never writes)

Several apps share one Cale. Cale decides who goes first with one rule that is the same on every computer (`docs/API.md`, "Who goes first"):
an app that asked for speed goes first, an app with no speed goal next, a `power: "light"` background job last.
On Windows and Linux a running background answer is paused for a foreground one. On the Mac a running answer is not stopped, but the foreground call starts next.
Test: `cale/tests/priority.py`. Live: start Background Sorter, then type in Quick Notes. The Hub shows the foreground app's first word stay near its goal.

## Mac numbers (M5, 24 GB, measured with `demo/check_apps.py`)

| App | Result |
|---|---|
| Quick Notes | first word 0.13 s alone; median 0.77 s while the Sorter runs (goal 1 s) |
| Private Doc Reader | 5,458 words read in 5.3 s, then first word 0.13 s per question, 7,800 tokens reused. Always the full size. |
| Meeting Recap | first word 0.7 s, whole recap and action items in under 10 s |
| Background Sorter | 20 of 20 files, 24 to 35 files per minute, receipts' amounts all correct |

## HP laptop numbers (8 CPU threads, 17 GB Windows / 8 GB WSL Linux, measured 2026-10-15 with `demo/check_apps.py`)

| App | Windows, CPU only | Windows, GPU reads prompts (MX150) | Linux (WSL), CPU only |
|---|---|---|---|
| Quick Notes | WELL, 13.3 tok/s | 12.4 tok/s, 1 of 3 calls missed | WELL, 10.4 tok/s |
| Private Doc Reader | WELL, read 7,824 tokens in 423 s | WELL, read in 63 s | WELL, read in 434 s |
| Meeting Recap | not measured (failed to start in that run) | WELL, 106 s | POOR, 137 s, first word 48 s |
| Background Sorter | 20 files in 278 s | 20 files in 251 s (POOR: memory 29.8% vs goal 28%) | 20 files in 185 s (POOR: memory goal 28% not reachable, used 59%) |

Files: `results/demo/hp-windows-cpu-sampling.json`, `hp-windows-gpu-fix.json`, `hp-linux-final.json`. Details and the GPU long-prompt fix are in `docs/research/adherence-sampling.md`.

## Known limits (say them before someone asks)

| Limit | Why |
|---|---|
| Chat needs Open WebUI installed first (about 2 GB, its own Python packages). | It is a real third-party app. It is pre-installed on the demo computers. |
| Pictures and sound work on the Mac only. | The Windows and Linux engine has no vision or audio files yet. |
| Doc Reader's time prediction is cautious on the Mac (said 12 s, took 5.6 s). | The prediction counts a cold start. |
| The model is a prototype (Gemma 3n). Its answers are not as good as the model Cale is meant to have. | The funding is for the Matryoshka model built for this. |
| On the old HP the GPU only reads prompts (it has 2 GB). Answers run on the CPU. | That is the fastest way measured on that card. A card that holds the whole model would also speed up answers. |
| Memory goals can miss by a few percent, and a small computer may not reach a low goal. The Scorecard says so (POOR). | Honest report, not a hidden trade-off. |
| The Allow question must be clicked by a person. | That is the point of it. |
| On the Mac a background answer that is already running is not stopped for a foreground app. | The Mac engine answers one at a time. A foreground call waits for the answer that runs now (a few seconds at most). |
