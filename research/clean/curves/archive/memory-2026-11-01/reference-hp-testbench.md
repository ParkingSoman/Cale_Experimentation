---
name: reference-hp-testbench
description: "The HP Windows laptop + WSL2 Ubuntu test bench: ssh commands, hp job script, heartbeats, traps, click-dialogs-there rule (merged)"
metadata:
  type: reference
---

Type: reference + feedback. Rules about the REAL Cale are in [[feedback-testing-rules]].

## Setup, commands and link fixes

Set up 2026-10-11 so Cale can be tested on Mac, Windows and Linux. The Windows laptop (`DESKTOP-TV573CG`, Windows 11 23H2, user `Shonu`, admin) runs an OpenSSH server. The Mac reaches it over Tailscale (100.95.121.88). Linux is Ubuntu 24.04 in WSL2 on the same laptop (user `cale`, kernel 6.18 microsoft-standard-WSL2).

| Want | Command |
|---|---|
| Windows (PowerShell) | `ssh hp '<cmd>'` or `scripts/remote_win.sh '<cmd>'` |
| Linux (Ubuntu) | `scripts/remote_linux.sh '<bash cmd>'` (copies the text to `C:\cale\last-linux-cmd.sh`, then runs it) |
| Reboot test | `ssh hp 'shutdown /r /t 5'`; SSH + Tailscale return with no login (tested) |

**Gotchas that cost time:**
- `systeminfo`, `Get-CimInstance Win32_Processor` and `wsl --list --online` hang or drop the SSH link. Use simple commands and registry reads.
- The link also drops for a moment while WSL starts or downloads. Never run a long job as the SSH command itself. Start it detached and write its log to `C:\cale\*.log`, then read the log with `ssh hp 'Get-Content C:\cale\x.log -Tail 3'`.
- PowerShell breaks nested quotes: never pass `bash -c "..."` through ssh. Copy a script file and run it.
- `wsl.exe` prints UTF-16: pipe through `tr -d '\0\r'`.
- WSL does not run as SYSTEM, so a SYSTEM scheduled task cannot install a distro. The install needs the normal user's own PowerShell.
- Do not create scheduled tasks, services or passwordless sudo on the laptop without asking: auto mode blocks them as persistence.
- Built-in `wsl.exe` was old. `wsl --update --web-download` fixed it.

**Fast loop (learned 2026-10-11, Phase 5 — cost about 2 hours before it was fixed):**
- Use `scripts/hp`: `hp run win|wsl NAME 'cmds'` starts a detached job, `hp wait NAME 300` is the 5-minute heartbeat, `hp log NAME`, `hp put FILE`. The job writes `C:\cale\jobs\NAME.log` and `NAME.done`.
- A child started with `Start-Process` over ssh DIES when the ssh session closes (sshd job object). Start jobs with WMI: `Invoke-CimMethod -ClassName Win32_Process -MethodName Create`. `hp` does this. It is a plain process, not a task or service.
- The Mac and the HP are both on Wi-Fi (one access point). A full-speed download on the HP stalls ssh for minutes. Cap HP downloads (`curl --limit-rate 20m`) or use Ethernet. Never move big files Mac→HP: scp dies at ~90 MB and runs ~10 MB/s. Let the HP download from Hugging Face itself, with `curl -C -` so it can resume.
- Copying from `/mnt/c` into WSL ext4 is ~2 MB/s. Download inside WSL instead.
- Do not start the same job twice: two copies kill each other's downloads.
- Antivirus (McAfee `mcshield`) showed no scan cost on a 1 GB write, so do not blame it without a test.

**Limits:** WSL2 proves the Linux code path, not every native Linux GPU driver. The HP has an Intel gen-9 iGPU, weak under WSL2.

See [[project-cale-phases-3-5]] for the Cale runtime that will be ported next, and the BlinkFix notes `windows-hp-harness` (older Chrome-based route to the same laptop).

**Link fixes (2026-10-13, Shonu: "should feel as breezy as the Mac"):**
- `~/.ssh/config` Host hp: NO shared connection (a half-dead master made calls hang 50 s), probe every 5 s, `ConnectTimeout 10`. `scripts/hp` `hssh` retries 4x when ssh exits 255. Jobs from `hp run win` run at BelowNormal priority.
- Calls cost about 0.5 s. During model loads the HP can stall tens of seconds (huge page-fault storm): a retry gets through. Do not shrink the ssh timeouts again.
- Wi-Fi card sleep disabled (`PnPCapabilities=24`). Do not run my own CPU-heavy monitors on the HP during benchmarks (they skew results).
- The official GPU backend DLLs load into our MinGW build. No Vulkan SDK is needed. A crash inside ctypes needs `CALE_CRASH_TRACE=1` plus `scripts/resolve_crash_offsets.py`.

## 5-minute heartbeats and traps (Shonu 2026-10-12)

Shonu (2026-10-12): the HP workflow is flaky, so report a **5-minute heartbeat** on every long HP job (`scripts/hp wait NAME 285`, then read the log and say what I see).

**Traps that cost time this session:**
- `scripts/hp run wsl NAME 'bash /mnt/c/cale/jobs/X.sh'` writes its OWN `NAME.sh`. If X equals NAME the job overwrites my script with a line that runs itself in a loop: it ate all of WSL's memory and killed the VM. Always give the script body a different name from the job.
  **I made this exact mistake THREE times (lx5, zoolin2, hugegrep).** Name the body `*_body.sh` / `*_body.ps1` and the job something else, every time, before running `hp run`. `wsl --terminate Ubuntu-24.04` stops a looping job.
- Never put a second `hp wait` after `hp run` in a way that hides a link drop: if `hp wait` prints "link down" or a timeout, loop `scripts/hp ping` until it is up, then wait again. The job keeps running on the HP.
- New code goes to the HP as a tarball: `tar czf code6.tgz cale nest scripts llama/*.py`, `scripts/hp put`, then the body script extracts it (`~/code6` in WSL, `C:\cale\code6` on Windows). Libraries: Linux `~/lbuild6/b/libcale_llama.so`, Windows `C:\cale\lbuild2\b_noomp\cale_llama.dll`; rebuild with `llama/build.sh` (1.5 min) and `llama/build.ps1` (30 s), then copy into `llama/dist/`.
- A `wsl --shutdown` leaves WSLg without a working X display (xdotool hangs). Do not shut WSL down unless needed.
- sshd on Windows cannot forward to WSL ports (forward to a Windows port instead; a small relay on Windows works). Easier: run the apps on the HP itself.
- The Mac Tailscale can be stopped: `tailscale up` brings it back, then `scripts/hp ping`.
- Real Allow dialog: Mac `scripts/click_allow_mac.sh`; Windows one-shot task (`schtasks /Create ... /IT`, run, then delete) with `scripts/click_allow_win.ps1`. User allowed clicking Cale's own dialog only, for the 2026-10-12 session.
- HP has python 3.12 (Windows venv `C:\cale\venv`, WSL `~/xplat-venv`), Node in `~/node/bin` (WSL) and `C:\cale\node\node-v22.11.0-win-x64`, no jq. No admin/sudo in WSL: use `apt-get download` + `dpkg -x` into `~/.local/pkgs`.

**How to apply:** Use `/private/tmp/claude-501/sshw.py` style helpers only with script FILES, never inline quotes through PowerShell. See [[reference-hp-testbench]].

**2026-10-14 traps:** (1) `scripts/hp put FILE` with a name that already exists in `C:\cale\jobs` can fail silently and the OLD file runs: use a new file name every time and never send put output to /dev/null. (2) cmake caches the source path: a new code folder cannot reuse an old build folder (build from the same code path, or a new build name). Put `set -o pipefail` and check the build exit code, `| tail` hides failures. (3) A stale suite job from an earlier session started again at WSL boot and ate the CPU during measurements: check `ps` in WSL for stray python before timing anything. (4) Killing a stray process I did not start is denied by the auto-mode classifier; killing my own job works. (5) tar on the Mac: put `--exclude` BEFORE the file list. (6) Mac `c++` lost its C++ headers after an update: build with `CMAKE_EXTRA="-DCMAKE_CXX_FLAGS=-isystem<SDK>/usr/include/c++/v1"`. (7) colima shares only $HOME with containers, not /private/tmp.

## The HP is my test bench (Shonu 2026-10-13)

Shonu (2026-10-13): "i need you to be able to click those linux allow dialogues. think of that computer as your test bench since i don't do any serious work on it, it's literally just to simulate my software on other OSes."

**Why:** testing Cale on other OSes is the HP's only job. Waiting for a person to click a dialog there blocks verification.

**How to apply:** click Cale's own dialogs on the HP myself. WSLg's display does not run in an SSH session, so use a virtual screen: `scripts/linux_allow_check.py` (Xvfb + zenity + xdotool, installed with `wsl -d Ubuntu-24.04 -u root -- apt-get install ...`; root works with no password). Install test tools in WSL freely. Still true: never pkill by pattern while a suite runs on the HP (it kills the suite's server; use pid files), and run long apt jobs as detached `hp run` jobs (an attached ssh drops). Related: [[reference-hp-testbench]], [[reference-hp-testbench]].
