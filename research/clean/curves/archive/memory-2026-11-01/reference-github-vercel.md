---
name: reference-github-vercel
description: "Private GitHub repo (which gh login, push rules) and Vercel team/site (rewrite demo, publish block, CLI route)"
metadata:
  type: reference
---

## GitHub
- Private repo `ParkingSoman/Cale_Experimentation` (https://github.com/ParkingSoman/Cale_Experimentation), created 2026-10-17. Remote `origin`. Branches `main` and `phase5-cross-platform` pushed.
- `gh` has two logins: `ParkingSoman` (Shonu's, use for Cale) and `omniya-bot` (omniya-lab org, different project). Check `gh auth status` before any write; switch with `gh auth switch`.
- Why: all Cale work backed up off the near-full shared disk. Push only when asked or at the end of a finished chunk. History scanned 2026-10-17: no keys/tokens, largest file 21 MB (GitHub blocks 100 MB). `.gitignore` skips `models/`, `dist/`, venvs, key files. Never add a model file, `cale-key`, `keys.json`, `hub.key`. See [[feedback-disk-hygiene]].

## Vercel
- Team "Shonu S's projects" (slug `shonu-ss-projects`, id `team_9QFzmMuYeIukRzzldQWnkaDQ`). Vercel MCP is connected but `get_auth_user` returns 404; `create_deployment` was refused by the auto-mode classifier ("Create Public Surface") on 2026-10-17 even after "do what you want with the test page". Do not route around it.
- Update 2026-10-18: Shonu logged in the Vercel CLI (`npx vercel login`, account `parkingsoman`). Site live at https://cale-rewrite.vercel.app (project `cale-rewrite`). MCP project creation stays blocked (403). Deploy with `npx vercel deploy --prod --yes --scope shonu-ss-projects` from a throwaway folder with the four files (`index.html`, `cale-web.js`, `get-cale.html`, `carrot.svg`); compare each live file with local byte for byte; test with Claude in Chrome using `?cale=http://127.0.0.1:<port>` and a TEST Cale so the real Cale shows no dialog ([[feedback-testing-rules]]). Chrome's local-network permission for that origin is already Allowed in his Chrome. Why: the Rewrite site needs a real public https address to test Chrome's local-network question and mixed content. Site files: `web/rewrite/`.
