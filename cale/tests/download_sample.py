"""The installer records how fast a download was (cale/install.py download, cale/samples.py). Run: python cale/tests/download_sample.py
A local server sends 52 MB. No model, no network.
"""
import http.server
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cale"))
home = Path(tempfile.mkdtemp(prefix="cale-dl-test-"))
os.environ["CALE_HOME"] = str(home)
import install  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:240]}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


BODY = os.urandom(1 << 20) * 52


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)

    def log_message(self, *a):
        pass


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
dest = home / "model.bin"
install.download(f"http://127.0.0.1:{srv.server_address[1]}/m", dest)
f = home / "samples.jsonl"
lines = [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []
check("a download sample is written", len(lines) == 1 and lines[0]["kind"] == "download", lines)
if lines:
    d = lines[0]
    check("it holds size, time, speed and the worst gap", d["mb"] >= 52 and d["seconds"] > 0 and d["mb_per_s"] > 0 and "worst_gap_s" in d, d)
    check("it holds numbers only", set(d) == {"t", "kind", "mb", "seconds", "mb_per_s", "worst_gap_s", "resumed", "capped"}, set(d))
srv.shutdown()
print("\nALL PASS" if not FAIL else f"\n{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
