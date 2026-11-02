# Config suggestions (2026-10-21). Nothing was edited.

Files checked: `~/.claude/CLAUDE.md` (540 B), `Cale_Experimentation/CLAUDE.md` (3 KB), `~/.claude/settings.json` (10 KB). No `Cale_Experimentation/.claude/` folder exists.

## Top suggestions
1. **Project CLAUDE.md is stale.** It says "Demo apps: Quick Notes plus ONE new second app". It is fine as a rule, but it carries the full demo decision table. Today the work is the ConvNeXt map track. Add one line at the top: "Research track: read research/clean/MAP-STATE.md". Move the demo table to `demo/README.md` (it already holds the run sheet) and keep only the 4 "Rules for suggestions".
2. **Hooks: 14 event types all call graphaas.** Most (Notification, PermissionDenied, StopFailure, TeammateIdle, Elicitation, ElicitationResult) run the same `hook notification` command. If you do not read graphaas output, turn these off. Fewer hooks means less noise at each turn and a faster Stop.
3. **Stop hook pile-up.** Three Stop hooks run each turn: `turn-commit.sh` (it makes the "wip: end of turn" commits you see in git log), graphaas stop, and `lesson-capture.sh`. The wip commits clutter history. Consider committing only on request, or squash before push.

## Other findings
- `lesson-rules.md` (memory) holds many rules from OTHER projects (GitHub Actions, CI, narrative design, omniya). They load for every Cale session. Move non-Cale rules to a global file or delete them. Rule L131 says "omit AI attribution footers". Your settings set `attribution` to empty, but this session's reminder asks for a `Claude-Session:` line. These two conflict. Pick one.
- `SessionStart` hook runs `BlinkFix-models/disk-guard.sh`. That is another project. Keep it only if the disk is still shared and near full.
- `permissions.allow` has only 7 entries and `skipDangerousModePermissionPrompt` plus `skipAutoPermissionPrompt` are on. The auto-mode classifier still blocks some things (Vercel deploy, prep scripts). Add explicit allow rules for the ones you want (for example `mcp__vercel__create_deployment`, as noted in memory).
- Plugins: `playwright` is enabled, but your rule is "Claude in Chrome only". Disable it so it cannot be picked by mistake. `code-simplifier`, `claude-md-management` and `security-guidance` add skills to every prompt. Disable any you do not use.
- Global CLAUDE.md is short and fine. One tip: the STE-100 rule fires for any answer over ~3 sentences. That is good, but it costs a skill load each time. Keep it.
- Many skills are listed in every session (anthropic-skills:*, omniya-lab, graphaas-planning). They fill the prompt. Disable the ones you never use.
- Old session folders in `~/.claude/projects/-Users-shonusengupta-Cale-Experimentation/` hold many `.jsonl` transcripts. They use disk. Archive transcripts older than 30 days if the disk is tight.
