# Meeting Recap

Paste a meeting transcript (or drop a .txt). The app streams a 3-5 bullet recap, then lists action items (owner, task, due) as a checklist.
It runs on Cale, a local AI runtime. Node 18+, no npm packages. Windows, Mac and Linux.

## Run

    node app.mjs

It prints `http://127.0.0.1:8824` and opens your browser. Click "Use sample" (loaded already) and "Write recap".

| Setting | Meaning |
|---|---|
| `CALE_URL` | Cale's address. Default `http://127.0.0.1:47821`. |
| `CALE_ENSURE=1` | Start Cale first if it is not running. This runs the `cale` command (from the Cale install). Off by default. |
| `NO_OPEN=1` | Do not open the browser. |

First run: Cale asks you to allow "Meeting Recap" (a dialog, or `cale allow "Meeting Recap"` in a terminal with no screen).
The key is kept in `data/cale-key`. If Cale rejects it, the app asks again.

## How it works

The browser cannot call Cale (Cale refuses requests with an `Origin` header). So `app.mjs` is a small local server. It serves the page
and calls Cale from Node, then streams the answer to the page (server-sent events).

- `recap` model object: `min_tokens_per_s: 3`, `max_tokens: 400`. Streamed.
- `actions` model object: `max_first_token_s: 20`, `max_tokens: 600`. Uses `response_format` `json_schema`.
- Both are leased objects. They end when you quit (Ctrl+C), or by themselves if the app dies.
- "Developer view" shows the Cale lines, read from `app.mjs` (the lines between `CALE-BEGIN` and `CALE-END`).

Footer: size Cale chose, seconds to first word, whole time, and whether the goals were met (from `cale.goals_actual`).
`cale-client.mjs` is Cale's public Node client, unchanged.

Set `CALE_NO_BROWSER=1` to start the app without opening a browser tab (the Cale Hub does this).
