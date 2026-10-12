"""Download the Cale model from Hugging Face and print progress lines an app can read.

usage: python fetch_model.py REPO DEST_FOLDER

Each progress line is:  CALE_PROGRESS {"stage": "model", "done": <bytes>, "total": <bytes>}
When every file is present at the right size, the file DEST/.complete is written.
An interrupted download resumes the next time (Hugging Face keeps the partial files).
"""
import json
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from tqdm.auto import tqdm


def main(repo, dest):
    dest = Path(dest)
    files = {s.rfilename: s.size for s in HfApi().model_info(repo, files_metadata=True).siblings}
    total, state = sum(files.values()), {"done": 0, "last": 0.0}

    def emit(force=False):
        now = time.time()
        if force or now - state["last"] > 0.25:
            state["last"] = now
            print("CALE_PROGRESS " + json.dumps({"stage": "model", "done": state["done"], "total": total}), flush=True)

    class Bar(tqdm):
        """Counts the bytes written to disk, over all files, into one number.

        snapshot_download makes two byte counters: one for network bytes ("Downloading bytes") and one
        for bytes written to disk ("Reconstructing..."). Counting both would double the total.
        A resumed file reports its bytes already on disk through update(), so no `initial` is needed.
        """

        def __init__(self, *a, **k):
            self._counts = k.get("unit") == "B" and str(k.get("desc", "")).startswith("Reconstructing")
            super().__init__(*a, **{**k, "disable": True})

        def update(self, n=1):
            if self._counts and n:
                state["done"] = min(total, state["done"] + int(n))
                emit()
            return None

    emit(True)
    snapshot_download(repo, local_dir=str(dest), tqdm_class=Bar)
    bad = [f for f, size in files.items() if not (dest / f).is_file() or (dest / f).stat().st_size != size]
    if bad:
        sys.exit(f"the download is incomplete or damaged: {', '.join(bad[:3])}")
    state["done"] = total
    emit(True)
    (dest / ".complete").write_text("ok\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
