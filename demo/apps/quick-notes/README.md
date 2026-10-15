# Quick Notes

A notes page with ghost-text autocomplete. Pause 400 ms while typing: Cale suggests about 20 tokens in grey. Tab accepts. Typing again cancels the old request. Under the editor: time to ghost text, size Cale chose, goal met or not. "Developer view" shows the Cale code (read live from `app.py`).

## Run
    python app.py            # Python 3.9+, standard library only
    CALE_URL=http://127.0.0.1:47899 python app.py     # other Cale address

Opens http://127.0.0.1:8821. Notes save to `data/notes.txt`; the Cale key to `data/cale-key`.

## Cale features used
- `POST /cale/connect` key, stored and reused (asked again on 401)
- Model object `autocomplete` (`max_first_token_s=1`, `max_tokens=20`) via `cale_client.model`
- `/v1/chat/completions` with that object; reads `X-Cale-Size` and `X-Cale-Goal-Met`
- Cancel: the app closes its HTTP connection to Cale, so Cale stops work
- `ensure_running()` if Cale is down

The browser cannot call Cale (Origin refused), so `app.py` is a small local server in between.

Set `CALE_NO_BROWSER=1` to start the app without opening a browser tab (the Cale Hub does this).
