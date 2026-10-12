"""App 2: Aider with Cale as its AI.

Aider itself is not changed. This launcher checks whether Cale is on this computer. If it is (and you did not pick
your own model), it asks once, gets a key, and starts Aider with two Cale model objects (ended when Aider quits):
  code    the main model (writes the code): a deadline for a long answer, and at most 40% of the RAM
  commit  the weak model (commit messages, summaries): a short deadline for a short answer
The launcher says what it needs. Cale picks the size per call, for this computer: the full size on a fast one,
a smaller size on a weaker one. Nothing is downloaded: Cale already holds the model.

Run:  <venv>/bin/python apps/aider/cale_aider.py [any aider arguments]        (Windows: <venv>/Scripts/python.exe)
Settings: CALE_URL (default http://127.0.0.1:47821). Goals: CALE_CODE_DEADLINE, CALE_CODE_TOKENS, CALE_MEMORY_PCT,
          CALE_COMMIT_DEADLINE, CALE_COMMIT_TOKENS
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

CALE_URL = os.environ.get("CALE_URL", "http://127.0.0.1:47821").rstrip("/")
STATE = Path(os.environ.get("CALE_AIDER_HOME", Path.home() / ".aider"))
KEY_FILE = STATE / "cale-key"
META_FILE = STATE / "cale-model-metadata.json"
APP_NAME = os.environ.get("CALE_APP_NAME", "Aider")                           # the name the Allow question shows
os.environ.setdefault("CALE_PORT", str(urlparse(CALE_URL).port or 47821))     # the cale package finds Cale at the same address
sys.path.insert(0, str(Path(os.environ.get("CALE_SRC", Path(__file__).resolve().parents[2]))))
# What Aider needs from a model, in its own words. Cale turns that into a size, per call, for this computer.
CODE = dict(name="code", deadline_s=float(os.environ.get("CALE_CODE_DEADLINE", "600")),
            max_tokens=int(os.environ.get("CALE_CODE_TOKENS", "1200")),
            memory_pct=float(os.environ.get("CALE_MEMORY_PCT", "40")))
COMMIT = dict(name="commit", deadline_s=float(os.environ.get("CALE_COMMIT_DEADLINE", "30")),
              max_tokens=int(os.environ.get("CALE_COMMIT_TOKENS", "60")))


def call(path, key=None, body=None, timeout=3):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(CALE_URL + path, data=None if body is None else json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def cale_running():
    try:
        return bool(call("/cale/status").get("ok"))
    except (urllib.error.URLError, OSError, ValueError):
        return False


def valid_key():
    try:
        key = KEY_FILE.read_text().strip()
        call("/v1/models", key)
        return key
    except (OSError, urllib.error.URLError, ValueError):
        return None


def get_key():
    print("Cale will ask you to click Allow on your screen.", flush=True)
    try:
        got = call("/cale/connect", body={"app": APP_NAME}, timeout=150)
    except urllib.error.HTTPError as e:
        why = json.loads(e.read() or b"{}").get("error", {}).get("message", f"Cale said {e.code}")
        sys.exit(f"Cale did not give Aider a key: {why}")
    STATE.mkdir(parents=True, exist_ok=True)
    KEY_FILE.write_text(got["api_key"])
    try:
        KEY_FILE.chmod(0o600)                       # no effect on Windows: the user's own folder is private there
    except OSError:
        pass
    return got["api_key"]


def write_metadata():
    """Tell Aider the size of the models and that they cost nothing, so it prints no warnings."""
    entry = {"max_input_tokens": 32768, "max_output_tokens": 4096, "input_cost_per_token": 0,
             "output_cost_per_token": 0, "litellm_provider": "openai", "mode": "chat"}
    STATE.mkdir(parents=True, exist_ok=True)
    META_FILE.write_text(json.dumps({f"openai/{CODE['name']}": entry, f"openai/{COMMIT['name']}": entry}))


def picked_a_model(argv):
    return any(a in ("--model", "-m") or a.startswith("--model=") for a in argv)


def has_cloud_key():
    keys = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY")
    return any(os.environ.get(k) for k in keys)


class keep_models:
    """Make Aider's two model objects while Aider runs. Aider works without them, so any failure is ignored."""

    def __init__(self, key):
        self.key, self.made = key, []

    def __enter__(self):
        try:
            from cale.client import model
            for spec in (CODE, COMMIT):
                m = model(CALE_URL + "/v1", self.key, **spec)
                m.__enter__()
                self.made.append(m)
            return [m.info for m in self.made]
        except Exception as e:
            print(f"(Cale could not make the models for Aider: {e}. Continuing without them.)", flush=True)
            self.close()
            return None

    def close(self):
        for m in self.made:
            m.close()
        self.made = []

    def __exit__(self, *exc):
        self.close()


def main():
    # Windows consoles use an old code page that cannot print the block character Aider draws. Aider then crashes
    # and waits at a bug-report prompt. UTF-8 output avoids that.
    os.environ.setdefault("PYTHONUTF8", "1")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass
    argv = sys.argv[1:]
    if not picked_a_model(argv) and (cale_running() or not has_cloud_key()):
        if not cale_running():
            try:
                from cale.client import ensure_running
                from cale import oslayer
                if (oslayer.home_dir() / "runtime" / "cale" / "server.py").exists():
                    print("Starting Cale...", flush=True)
                    ensure_running()
            except Exception as e:
                print(f"(Cale could not start: {e})", flush=True)
        if not cale_running():
            sys.exit("Cale (AI that runs on this computer) is not installed. Open an app that includes Cale, such as the "
                     "Open WebUI demo, and click Get Cale. Aider will then find it. Or run: python cale/install.py")
        key = valid_key()
        if not key:
            if sys.stdin.isatty() and "--yes-always" not in argv:
                try:
                    answer = input("Cale (AI that runs on this computer) is installed. Use it for Aider? [Y/n] ").strip().lower()
                except EOFError:                            # no one is there to answer: do not decide for them
                    answer = "n"
                if answer in ("n", "no"):
                    sys.exit("OK. Pass --model and an API key to use another AI.")
            key = get_key()
        write_metadata()
        os.environ["OPENAI_API_BASE"] = CALE_URL + "/v1"
        os.environ["OPENAI_API_KEY"] = key
        for other in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY"):
            os.environ.pop(other, None)                     # Aider must not pick another provider by itself
        argv = ["--model", f"openai/{CODE['name']}", "--weak-model", f"openai/{COMMIT['name']}", "--edit-format", "whole",
                "--model-metadata-file", str(META_FILE), "--no-show-model-warnings", "--no-check-update",
                "--analytics-disable"] + argv
        print(f"Aider is using Cale: '{CODE['name']}' writes code, '{COMMIT['name']}' writes commit messages. "
              "Cale picks the size for this computer.", flush=True)
        from aider.main import main as aider_main
        with keep_models(key) as infos:
            if infos:
                h = infos[0]
                print(f"Cale holds the {h['size_pct']}% size in memory for Aider "
                      f"({h['ram_gb']} GB, goal {'met' if h.get('goal_met') else 'NOT met'}).", flush=True)
            code = aider_main(argv)
        sys.exit(code)
    from aider.main import main as aider_main
    sys.exit(aider_main(argv))


if __name__ == "__main__":
    main()
