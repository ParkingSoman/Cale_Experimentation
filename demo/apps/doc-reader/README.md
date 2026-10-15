# Private Doc Reader

Ask questions about a long private document. Answers always use Cale's full size (`gemma-3n:smart`).

Run (Python 3.9+, standard library only):

    python app.py

It opens http://127.0.0.1:8822. Cale's address comes from `CALE_URL` (default `http://127.0.0.1:47821`).
The first run asks Cale for a key; allow the app when your computer asks. The key is kept in `data/cale-key`.

How it works:
1. You add a .txt or .md file (or paste text, or try `sample/lease.md`).
2. The app asks Cale `POST /cale/predict` and says how long reading will take on this computer.
3. One short call makes Cale read the document. Later questions extend the same chat, so Cale reuses the reading (prompt cache).
4. Answers stream. The footer shows size, first-word time, tokens/s and tokens reused.

If the document is over Cale's context limit, the app says so and offers the first part.
`cale_client.py` is Cale's public client, copied unchanged. The browser cannot call Cale, so `app.py` does it.
