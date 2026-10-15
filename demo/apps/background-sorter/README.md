# Background Sorter

Sorts a folder of .txt notes, receipts and emails into one table, quietly, using Cale.
Python 3.9+, standard library only.

    python app.py                      # Cale at http://127.0.0.1:47821
    CALE_URL=http://127.0.0.1:47899 python app.py

The page opens at http://127.0.0.1:8823. The app's own server calls Cale (a browser cannot).

- Pause / Resume: the file being sorted finishes first.
- Folder: type a path (default `sample/`). Only top-level `.txt` files.
- Results: `data/sorted.csv`, `data/sorted.json`. Done files are skipped on restart; changed files are redone.
- Key: `data/cale-key`. Delete it to ask Cale for a new one.
- Cale settings: model object `sorter`, `power=light`, `memory_pct=28`, `max_tokens=220`, `on_miss=best_effort`
  (never refuses, never waits in line for others). Busy or refused answers (422, 429, 507, 500) mean: wait, retry, show a plain message.
- `cale_client.py` is Cale's public Python client, unchanged.

Set `CALE_NO_BROWSER=1` to start the app without opening a browser tab (the Cale Hub does this).
