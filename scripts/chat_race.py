"""Open the chat cold and send one message the moment it says READY. Prints the timeline. Works on Mac, Windows and Linux.

It starts its OWN test Cale (port 47899, its own home folder) and its own chat. It never touches the installed Cale (port 47821).

usage: python scripts/chat_race.py HOME_DIR --bundle CHAT_BUNDLE [--node NODE] [--message TEXT] [--timeout 600]
  HOME_DIR  a new, empty folder for the test Cale
  --bundle  the chat build (dist/chat on Mac, dist/chat-win on Windows)
  --node    Cale's own Node (the newest Node may break the chat). Default: CALE_NODE, then the node on PATH.
Model and engine come from the environment, as for scripts/cale_dev_server.py (CALE_MODEL, CALE_BACKEND, CALE_LLAMA_LIB).
Output: when the chat said READY, when the first word arrived, when the answer ended, and every step from cale/timeline.py.
"""
import argparse
import http.cookiejar
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = 47899


def stop(p):
    """Stop a process we started, and the programs it started."""
    if p.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
    else:
        p.terminate()
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("home")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--node", default=os.environ.get("CALE_NODE") or shutil.which("node"))
    ap.add_argument("--message", default="Say hello in five words.")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--reuse-home", action="store_true", help="the folder may hold an earlier run of this test (to measure a second start). Its timeline is cleared.")
    ap.add_argument("--feel", action="store_true", help="after the first message, move the Quicker/Smarter slider to five positions and send a message at each: prints the size Cale used")
    ap.add_argument("--docs", action="store_true", help="after the first message, add a small document (a made-up fact), let the chat read it, and ask about it: checks the document reader, the embedding and the vector database")
    ap.add_argument("--cold", action="store_true", help="do not start the test Cale first: the chat starts it, as when a person opens Cale and Cale is not running")
    ap.add_argument("--wait-before-chat", type=float, default=5, help="seconds between the test Cale starting and the chat starting")
    a = ap.parse_args()
    home = Path(a.home).resolve()
    if home.exists() and any(home.iterdir()) and not (a.reuse_home and (home / "server.out").exists()):
        sys.exit(f"{home} is not empty. Give a new folder, so no real data is touched. (--reuse-home works only on a folder this test made.)")
    home.mkdir(parents=True, exist_ok=True)
    (home / "timeline.jsonl").unlink(missing_ok=True)
    data = Path(tempfile.mkdtemp(prefix="cale-chatdata-"))
    base_env = {**os.environ, "CALE_HOME": str(home), "CALE_PORT": str(PORT), "CALE_DEBUG": "1", "CALE_AUTO_APPROVE": "1",
                "CALE_IDLE_S": "900", "PYTHONUNBUFFERED": "1", "CALE_NO_PATH": "1", "CALE_NO_LAUNCH_AGENT": "1"}
    base_env.pop("CALE_RUN", None)
    base_env.setdefault("CALE_MODEL", str(ROOT / "models" / "gemma-3n-e4b-4bit"))      # as scripts/cale_dev_server.py does (a cold start has no such script)
    py = sys.executable
    server = None
    if a.cold:
        (home / "server.out").write_text("(cold: the chat starts the test Cale; its log is logs/cale.log)\n")
    else:
        server_log = open(home / "server.out", "w")
        server = subprocess.Popen([py, str(ROOT / "scripts" / "cale_dev_server.py"), str(home)], cwd=ROOT, env=base_env, stdout=server_log, stderr=subprocess.STDOUT)
    chat = None
    try:
        if server:
            end = time.time() + 60
            while time.time() < end:
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{PORT}/cale/status", timeout=2).read()
                    break
                except Exception:
                    time.sleep(0.3)
            else:
                sys.exit("the test Cale did not start; see " + str(home / "server.out"))
            time.sleep(a.wait_before_chat)
        chat_env = {**base_env, "CALE_NODE": a.node or "", "CALE_CHAT_BUNDLE": str(Path(a.bundle).resolve()), "CALE_CHAT_DATA": str(data)}
        out = open(home / "chat.out", "w")
        chat = subprocess.Popen([py, "-m", "cale.chat"], cwd=ROOT, env=chat_env, stdout=out, stderr=subprocess.STDOUT)
        t_chat = time.time()
        ready = None
        while time.time() - t_chat < a.timeout:
            m = re.search(r"READY (\d+) (\S+)", (home / "chat.out").read_text(errors="replace"))
            if m:
                ready = (time.time(), int(m.group(1)), m.group(2))
                break
            if chat.poll() is not None:
                sys.exit("the chat stopped before READY; see " + str(home / "chat.out"))
            time.sleep(0.05)
        if not ready:
            sys.exit("no READY in time")
        t_ready, port, token = ready
        jar = http.cookiejar.CookieJar()
        op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        op.open(f"http://127.0.0.1:{port}/?t={token}", timeout=30).read()
        t_send = time.time()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/workspace/my-chats/stream-chat", data=json.dumps({"message": a.message, "mode": "chat"}).encode(),
                                     headers={"Content-Type": "application/json"})
        first, text, err = None, "", False
        with op.open(req, timeout=a.timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    d = json.loads(line[5:])
                except ValueError:
                    continue
                if d.get("error"):
                    err = d["error"]
                if d.get("textResponse"):
                    first = first or time.time()
                    text += d["textResponse"]
        t_end = time.time()
        docs = None
        if a.docs:
            docs = {}
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                import doc_samples
                cases = doc_samples.samples()
                boundary = "----caleboundary7d1"
                for name, mime, body, question, want in cases:
                    try:
                        res = {}
                        mp = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\nContent-Type: {mime}\r\n\r\n").encode() + body + f"\r\n--{boundary}--\r\n".encode()
                        t1 = time.time()
                        up = urllib.request.Request(f"http://127.0.0.1:{port}/api/workspace/my-chats/upload-and-embed", data=mp, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
                        raw_up = op.open(up, timeout=300).read()
                        try:
                            r = json.loads(raw_up)
                        except ValueError:
                            raise RuntimeError("the upload answered: " + raw_up[:100].decode("utf-8", "replace"))
                        res["embed_s"] = round(time.time() - t1, 1)
                        res["embedded"] = bool(r.get("success"))
                        if not res["embedded"]:
                            res["why"] = str(r.get("error") or r)[:160]
                        else:
                            q = urllib.request.Request(f"http://127.0.0.1:{port}/api/workspace/my-chats/stream-chat", data=json.dumps({"message": question, "mode": "chat"}).encode(), headers={"Content-Type": "application/json"})
                            ans = ""
                            with op.open(q, timeout=a.timeout) as rr:
                                for raw in rr:
                                    line = raw.decode("utf-8", "replace").strip()
                                    if line.startswith("data:"):
                                        try:
                                            ans += json.loads(line[5:]).get("textResponse") or ""
                                        except ValueError:
                                            pass
                            res["answer"] = ans.strip()[:80]
                            res["knows_the_fact"] = want in ans.lower()

                    except Exception as e:
                        res = {**res, "error": f"{type(e).__name__}: {str(e)[:100]}"}
                    docs[name] = res
                docs["all_ok"] = all(v.get("knows_the_fact") for v in docs.values() if isinstance(v, dict))
            except Exception as e:
                docs["error"] = f"{type(e).__name__}: {str(e)[:150]}"
        sweep = []
        if a.feel:
            key = (home / "chat.key").read_text().strip()
            for pos in (0.0, 0.25, 0.5, 0.75, 1.0):
                fr = urllib.request.Request(f"http://127.0.0.1:{port}/cale-feel", data=json.dumps({"position": pos}).encode(),
                                            headers={"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{port}"})
                try:
                    op.open(fr, timeout=60).read()
                except Exception as e:
                    sweep.append({"position": pos, "error": f"slider: {e}"})
                    continue
                cr = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
                                            data=json.dumps({"model": "chat", "max_tokens": 48, "temperature": 0, "messages": [{"role": "user", "content": "Count from one to twenty in words."}]}).encode())
                t1 = time.time()
                try:
                    with urllib.request.urlopen(cr, timeout=a.timeout) as r:
                        body = json.loads(r.read())
                        n = (body.get("usage") or {}).get("completion_tokens") or 0
                        sweep.append({"position": pos, "size_pct": r.headers.get("X-Cale-Size"), "goal_met": r.headers.get("X-Cale-Goal-Met"),
                                      "predicted_tps": r.headers.get("X-Cale-Predicted-Tokens-Per-S"), "seconds": round(time.time() - t1, 1), "tokens": n})
                except Exception as e:
                    sweep.append({"position": pos, "error": str(e)[:80]})
        time.sleep(0.5)
        marks = []
        tl = home / "timeline.jsonl"
        if tl.exists():
            for l in tl.read_text().splitlines():
                try:
                    d = json.loads(l)
                except ValueError:
                    continue
                if d.get("kind") == "mark":
                    marks.append((d["t"], d["step"], d.get("seconds")))
        marks += [(t_ready, "** CHAT READY (window can draw)", None), (t_send, "** MESSAGE SENT", None)]
        if first:
            marks.append((first, "** FIRST WORD ON SCREEN", None))
        marks.append((t_end, "** ANSWER DONE", None))
        t0 = next((t for t, s, _ in sorted(marks) if s == "chat_start"), t_chat)
        print(f"{'step':36s} {'since chat start':>16s}")
        for t, s, sec in sorted(marks):
            if t >= t0 - 0.01:
                print(f"{s:36s} {t - t0:>15.1f}s" + (f"   ({sec} s)" if sec is not None else ""))
        print(f"\nanswer: {text.strip()!r}   error: {err}")
        if docs is not None:
            print("\ndocument check:", json.dumps(docs))
        if sweep:
            print("\nslider position -> what Cale used")
            for e in sweep:
                print("  ", json.dumps(e))
            sizes = [int(float(e["size_pct"])) for e in sweep if e.get("size_pct")]
            print(f"   sizes used: {sizes}  (a slider that works moves from small to large)")
        print(json.dumps({"ready_s": round(t_ready - t0, 2), "first_word_s": round(first - t0, 2) if first else None,
                          "done_s": round(t_end - t0, 2), "message_to_first_word_s": round(first - t_send, 2) if first else None,
                          "ok": bool(text.strip()) and not err}))
        return 0 if text.strip() and not err else 1
    finally:
        if chat:
            stop(chat)
        if server:
            stop(server)
        else:                                                         # the chat started the test Cale (cale/service.py): stop it by its pid file
            subprocess.run([py, "-c", "from cale import service; service.stop()"], cwd=ROOT, env=base_env, capture_output=True)
        shutil.rmtree(data, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
