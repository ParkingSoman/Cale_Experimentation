# The consumer demo (2026-10-16): read this section first

The demo is about what a consumer sees. Nothing technical is on any screen the audience sees. The presenter types nothing.

| Part | What the consumer does | Files |
|---|---|---|
| 1. Cale, once | Double-click the Cale download. A plain page says "Setting up", then "You are all set". | Mac `dist/Cale-mac.dmg`, Windows `Cale-Setup.exe` |
| 2. An app | Double-click an app download. It opens in the browser and works. | Mac `dist/apps/Quick-Notes-mac.dmg`, `dist/apps/Rewrite-mac.dmg`. Windows `Quick-Notes-Setup.exe`, `Rewrite-Setup.exe` |
| 3. Another app | Double-click the second app. No second Cale download. | same |

Apps are small (about 100 KB on Mac, 0.2 MB on Windows). They do not carry Cale. Each app finds the installed Cale and runs with the Python that Cale installed.
If Cale is missing, the app says so in plain words and offers to get it (it looks for the Cale download in Downloads and on the Desktop).

The two demo apps:

| App | Port | Kind of app | What it asks Cale for |
|---|---|---|---|
| Quick Notes | 8821 | grey ghost text while you type | first word in about a second |
| Rewrite | 8825 | pick a tone, new text streams in | words faster than you read (6 tokens/s) |

Say out loud, with no screen showing it: "Same app code on both computers. Each app asks for a feel. Cale picks the model size that gives that feel. So both apps feel the same on a new Mac and a ten-year-old laptop."
The technical view of every app and of Cale is hidden. Add `?dev=1` to the address to see it (for the funding pitch only).

How to build (on the Mac): `python3 scripts/build_installer.py mac`, `python3 scripts/build_app.py mac all`. For Windows: `python3 scripts/build_installer.py win`, `python3 scripts/build_app.py win all`, copy the folders to the HP, run `build_exe.ps1` and `build_app_exe.ps1` there (they sign with the demo certificate).
Warnings: a file built on this Mac has no quarantine mark and opens with no warning. A browser download does get the warning (the install is shown as a recorded time-lapse). The HP was prepared with `prep_windows_demo.ps1` on 2026-10-16 (undo: `-Undo`).

Measured on 2026-10-16 (real `.exe` files, fresh Cale install on the HP, test mode for the Allow question):

| Computer | Quick Notes first word | Rewrite |
|---|---|---|
| Mac (M5) | 0.2 to 0.4 s, size 100% | whole rewrite 0.7 s, size 100% |
| HP Windows | 0.84 to 1.29 s, size 95 to 100% (4 of 6 under 1 s) | 3.7 to 4.1 s, size 90 to 95%, speed goal met |

Mac dry run (2026-10-16): installed `Cale-mac.dmg` into a fresh test folder (about 2 minutes), then opened both apps from their own `.dmg` files. Quick Notes first word 0.1 s, Rewrite 100% size, goals met. Your real Cale was not touched.

HP clean dry run (2026-10-16, final files, fresh folder, real click on the Allow question): Cale installed through `Cale-Setup.exe` (served the plain page). Both apps opened from their own `.exe`, and a click on each Allow question worked (clicked by the test helper, as a person would). Quick Notes first word 0.88 s, size 100%, goal met. Rewrite 3.5 s in total, size 95%, goal met. The set-up step took 23 minutes this time and 9 minutes the first time (cause not found: the HP was busy; this is why the install is a time-lapse).

Known gaps: the HP set-up step "fitting to this computer" took about 9 minutes (fine for a time-lapse). Linux has no consumer double-click path (the HP Linux is inside WSL), so it is covered by the scorecard and by speaking. The plain Cale page, the "needs Cale" dialogs and the Allow wording were not looked at by eye (the Chrome extension was not connected).

---

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
