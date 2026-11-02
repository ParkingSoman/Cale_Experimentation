# Why the old laptop is slow at things that are not the model (2026-10-18)

The question: the sameness goal fails on chat start (Mac 1.3 s, HP 5.7 s) and on install (Mac 32 s, HP 384 s). Both computers are on the same internet. Where does the gap come from, if it is not the model?

## Measured, not guessed

`python scripts/machine_probe.py` on both computers (results: `results/machine-probe-mac.json`, `results/machine-probe-hp.json`).

| Cost | Mac (M5) | HP (i7-8550U, NVMe, two virus scanners) | HP slower by |
|---|---|---|---|
| Internet, 33 MB from a public server | 28.4 MB/s | 28.9 MB/s | **1.0x** (the same) |
| One CPU thread (Python loop) | 0.088 s | 0.627 s | 7.1x |
| Start a Python program | 0.011 s | 0.077 s | 7.0x |
| Import Cale's server | 0.041 s | 0.309 s | 7.5x |
| Start Node | 0.025 s | 0.083 s | 3.3x |
| Create small files | 21,761 per s | 217 per s | **100x** |
| Read small files | 63,774 per s | 2,264 per s | 28x |
| Copy small files | 9,963 per s | 586 per s | 17x |
| Delete small files | 36,024 per s | 2,880 per s | 12.5x |
| Write a big file (with flush) | 11.9 GB/s | 0.65 GB/s | 18x |

The HP disk is an NVMe SSD (not the problem). The HP runs **Windows Defender and McAfee at the same time**: every file that is made or opened is checked twice.

## What this says

1. **The network is not the gap.** Downloads take the same time on both. (The model download is 4.4 GB on Windows and 5.8 GB on the Mac: about the same time.)
2. **Work that is CPU-bound is 3 to 7x slower on the HP.** Starting Python or Node, importing, parsing JavaScript. This is the low end of the range. No code change makes a 2017 laptop thread as fast as an M5 thread. The only way to make a CPU-bound step feel the same is to **do less of it at the moment a person waits**.
3. **Work that touches many files is 12 to 100x slower on the HP.** The chat is tens of thousands of files. Unpacking them (209 s measured), copying them (150 s), and starting the chat server (it opened about 2,600 files: 4.7 s against 1 s) all pay this. This is the biggest and most avoidable gap: it is not about speed of the computer, it is about how many files we ask it to handle.
4. **The model is a separate gap**, and it is the part the Matryoshka work is for: the same model runs 6.7x slower on the HP (5.6 against 35 tokens/s on the full size). The hardware test, now in fixed time, and the relative goals cover it.

## What was done (and what it gave)

| Change | Effect (measured) |
|---|---|
| Pack the chat server and the document reader into one file each, the jobs into one shared pack (`scripts/pack_chat.mjs`) | The chat server opens one file at start instead of about 2,600 |
| Remove the files nothing can reach, keep the license texts in one file (`scripts/chat_prune.py`) | Chat files: 61,000 to 14,064 (Mac) and 12,194 (Windows). Unpack on the HP: 251 s to 56 s. Mac installer: 559 to 385 MB |
| The window never waits for the AI | Chat ready on the HP: 40 to 75 s before, then 5.7 s, now 2.3 to 3.3 s (Mac 0.9 to 1.3 s). The document reader now starts after the chat is ready. |
| Hardware test in fixed time (`cale/tuning.py`, `scripts/tune_check.py`) | Mac 131 s to 23 s, the same in every run. HP 311 s to 64 to 119 s. Mac predictions within 1 to 2% of real answers |
| Placement test (CPU against graphics card): lighter probe, ways dropped sooner, a time budget | HP with a card: about 295 s to about 234 s |
| Heat learned in use, not measured at install | 120 s less |

## Blue-sky ideas, by what they attack

Ordered by how much of the measured gap each could remove. None of them needs the HP to be faster.

| # | Idea | Attacks | Rough size of the win | Cost or risk |
|---|---|---|---|---|
| 1 | **Do not start at the moment of opening: keep it running.** Keep the chat server alive after the window closes (hours, not minutes), and on Windows start Cale at login (the Mac already does). Opening is then connecting: about 0.1 s on both computers. | CPU and files at open | Chat ready 5.7 s to under 1 s on the HP, after the first start of the day | A background program the person did not ask for (Windows). Memory used while idle (about 300 MB for the chat server). **Your decision.** |
| 2 | **Start from a snapshot, not from files.** Node can save the state of a started program (a V8 startup snapshot) and start from it: no JavaScript to read or parse. | Files and CPU at start | The chat server start could fall to tens of milliseconds on both | Build step per Node version and CPU; needs the pack to avoid loading files at start; native add-ons must be loaded after the snapshot |
| 3 | **Ship far fewer files.** Today about 28,000. The needed set is the native add-ons (sharp, onnxruntime, lancedb, prisma, canvas), the front-end page (about 250 files), the packs, and the libraries that must stay as files (the embedding library). One shared pack for the background jobs would let about 600 more packages go (about 15,000 files, about 70 s of unpack on the HP). | Files (unpack, copy, open) | Unpack on the HP from 209 s to under 40 s | Needs a careful test of every feature the packs can no longer find on disk |
| 4 | **One process instead of five.** The chat launcher (Python), the Cale server (Python), the chat server (Node), the document reader (Node) and the window are separate programs. Starting a program is 3 to 7x slower on the HP. Start the document reader only when the first document arrives; fold the launcher into the Cale server. | CPU, process start | Chat ready about 1 s sooner on the HP | Moderate code change in `cale/chat.py` |
| 5 | **Python from one zip.** Python reads one zip file instead of hundreds of module files (`zipimport`). Import of the server was 0.31 s on the HP. | Files and CPU | About 0.1 to 0.2 s per Python start | Small. Third-party packages (numpy) still load from files |
| 6 | **Sign the programs.** Antivirus scanners treat signed files from known publishers more lightly, and Windows stops warning. | Scanner cost | Unknown: needs a test with a signed copy | A code-signing certificate (about 100 to 400 USD a year). Already needed for a public release |
| 7 | **Do the per-model work once, at build time.** The repack of the model file (59 to 64 s on the HP) does not depend on the computer. | Install | 60 s less on Windows and Linux | Cale hosts the prepared file: hosting cost. **Your decision.** |
| 8 | **Tier the app's own startup like the model: essential, then deferred.** Show the window at once (feedback), make typing possible with the minimum, and load the rest later in a fixed order. A slower computer defers more. | Everything the person waits for | Makes usable time depend on the essential set only | Needs a clear list of what is essential |
| 9 | **Fixed-time everything that can be measured.** Done for the hardware test. The same idea for the install: show the true state of each step against a clock, and do steps in parallel when they use different resources (network, disk, CPU). | Install | Install under the sum of its parts | Done in part: the Windows model download now starts first |

## What is left after all of that

The chat is ready 2.5 times later on the HP than on the Mac (3.0 to 3.3 s against 1.2 s): about the CPU ratio of the two computers (3 to 7x), so the remaining lever is to do less at the moment of opening (ideas 1, 2, 4). And the model: 6.7x slower answers on the HP at the smallest size that fits. That is the range Matryoshka research must widen, and it is the one gap the plan accepts.

## Update (late 2026-10-18): two costs that were pure Windows, not CPU

| Finding | Measured | Fix |
|---|---|---|
| A connection to a closed local port takes 2.0 s to be refused on the HP (6 ms on the Mac). Every "is it up yet?" poll paid it. | `scripts` probe: 2.03 s, 2.04 s, 2.02 s | The chat server writes a file when it listens. The launcher waits for the file. "Is Cale up?" probes with a 0.25 s wait. |
| Node compile cache never filled on Windows (the servers are ended by force) | cache folder empty after 4 runs; 311 files after the flush fix | `preload.js` flushes the cache 4 s after the start. |

Chat ready: Mac 0.4 s, HP 1.55 s (was 1.3 s and 3.0 s). The idea "keep the chat running" was dropped by decision.
