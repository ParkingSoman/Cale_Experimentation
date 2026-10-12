# Hidden app zoo: baseline before Phase 6 (2026-10-12)

12 probe apps written by a reviewer agent that read only `docs/` (no Cale code). Run on three systems against Cale at its
Phase 5 state. Raw lines: `mac.txt`, `hp-linux.txt`, `hp-windows.txt`. The probe apps live outside the repo (`~/Cale_Zoo`).

| App | Mac (MLX) | HP Linux (CPU) | HP Windows (CPU) | What the app needed |
|---|---|---|---|---|
| agent-tools | FAILS | FAILS | FAILS | `tools` is accepted (200) and ignored |
| json-extract | FAILS | FAILS | FAILS | `response_format` ignored, reply is prose |
| long-doc-qa | WELL (10k tokens, first token 16 s) | FAILS | FAILS | HP: 5k prompt times out (slow prompt reading), 10k prompt gives 500 (context fixed at 8192) |
| summarizer | WELL (30 s) | FAILS (839 s, 1 of 3 facts) | FAILS (836 s, 1 of 3 facts) | The HP reads prompts so slowly that 4 chunk calls take 14 minutes |
| two-apps-at-once | POOR | POOR* | POOR* | Header says goal met while the app waited 7-10x its goal (queue wait not counted) |
| node-app | WELL | WELL | WELL | nothing |
| curl-script | WELL | WELL | WELL | nothing |
| streaming-backend | WELL (34 chunks/s) | WELL (6.8) | WELL (4.2) | nothing, but Windows is 38% slower than Linux on the same CPU |
| low-memory-app | POOR | POOR | POOR | memory goal missed or met on paper, call still runs at 100% size |
| battery-background-job | FAILS | FAILS | FAILS | no way to say "run gently" |
| stop-and-sampling | FAILS | FAILS | FAILS | `stop`, `seed`, `top_p` accepted and ignored |
| image-input | FAILS (honest 400) | FAILS (honest 400) | FAILS (honest 400) | image input |

`*` The checker printed WORKS_WELL for these two before it was fixed. Read by hand: the 4 s goal app waited 30 s (Linux) and 39 s (Windows).

Score (apps that work well x systems that run them): Mac 5/12, HP Linux 3/12, HP Windows 3/12 = 11 of 36 cells.

## Findings that the apps exposed

1. Silent acceptance: unsupported request fields get a 200 and are ignored (tools, response_format, stop, seed, top_p, unknown fields). This breaks PRINCIPLES rule 8.
2. One oversized prompt (45k tokens) left the Mac engine broken until restart (`There is no Stream(gpu, ...)`), and `/cale/status` still said ok.
3. The same app behaves differently per computer: the Mac accepts 10k-token prompts, the HP gives 500 (context 8192), and the error is not the documented 413.
4. Goal reporting lies under load: `X-Cale-Goal-Met: true` while the queue wait made the app miss its deadline.
5. A memory goal can say "48%" while chat calls run at 100%.
6. On the HP, prompt reading is the bottleneck, not answer speed. The deadline-in-seconds goal cannot describe that.
7. Windows ran 38% slower than Linux on the same CPU (streaming 4.2 vs 6.8 chunks/s).
8. The Linux Allow dialog tool (zenity) may never close. Cale raised TimeoutExpired and returned a 500. Fixed in `464ceea`.

## After stage 2a (2026-10-12, same 12 apps)

| System | Works well before | Works well after |
|---|---|---|
| Mac | 5 | 8 |
| HP Linux | 3 | 5 |
| HP Windows | 3 | 5 |
| **Total (of 36)** | **11** | **18** |

Rule: an app "works well" only if it meets the goals it states. Cale now says honestly when it misses a goal, and that is better
than before, but a 4-second goal that waited 32 seconds is not "well" (the zoo's own check printed WORKS_WELL for
`two-apps-at-once` on both HP systems because the header said `goal_met=false`: counted here as POOR). The first count of 20 was too generous.

Raw lines: `mac-after-2a.txt`, `hp-linux-after-2a.txt`, `hp-windows-after-2a.txt`.
What still fails: native tool calls and JSON schema (stage 6, now an honest 400), image input (stage 6), "run gently" jobs (stage 4),
and on the slow HP: the long document and the summarizer (prompt reading at 15 tok/s: GPU or a smaller context-reading cost).
