"""How accurate are Cale's speed predictions? Starts a fresh test Cale (own home, so nothing is learned yet) and compares.

usage: python scripts/predict_check.py MODEL [--lib LIB] [--load N] [--sizes e4b,e2.98-block,e2b] [--tokens 64]

1. before any answer: predicted answer speed and first-token time for every size ("estimate")
2. one answer at the full size. Compare. This is the *first estimate* error.
3. after that answer: predictions again ("learned"). Then one answer at each other size. This is the *after the first answer* error.
--load N starts N busy programs first (CPU on Windows and Linux, a second model on a Mac's GPU is not used here: CPU only).
Gate (docs/PLAN.md): first estimate within +-25%, after the first answer within +-10%.
"""
import argparse
import json
import multiprocessing
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:47899"
PROMPT = ("You are a careful assistant. Read this text about the sea and then write a long, detailed essay about it. " +
          "The sea covers most of the planet and moves heat around the world. " * 22)


def call(path, body=None, key=None, timeout=900):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_closed(timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        try:
            socket.create_connection(("127.0.0.1", 47899), 1).close()
            time.sleep(0.5)
        except OSError:
            return


def wait_ready(timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        try:
            call("/cale/status")
            return
        except Exception:
            time.sleep(1)
    raise SystemExit("the test server did not start")


def burn(stop):
    x = 0
    while not stop.is_set():
        x += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lib")
    ap.add_argument("--load", type=int, default=0)
    ap.add_argument("--sizes", default="e4b,e2.98-block,e2b")
    ap.add_argument("--tokens", type=int, default=64)
    a = ap.parse_args()
    home = tempfile.mkdtemp(prefix="cale-pred-")
    env = dict(os.environ, CALE_MODEL=a.model, PYTHONUNBUFFERED="1", PYTHONPATH=str(ROOT))
    if a.lib:
        env["CALE_LLAMA_LIB"] = a.lib
    srv = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "cale_dev_server.py"), home], cwd=ROOT, env=env,
                           stdout=open(Path(home) / "server.log", "w"), stderr=subprocess.STDOUT)
    stop, burners = multiprocessing.Event(), []
    try:
        wait_ready()
        key = call("/cale/connect", {"app": "predict-check"})["api_key"]
        sys.path.insert(0, str(ROOT))
        from nest.levels3n import load_levels
        pct = {l["id"]: l["pct"] for l in load_levels()}
        if a.load:
            burners = [multiprocessing.Process(target=burn, args=(stop,)) for _ in range(a.load)]
            [b.start() for b in burners]
            time.sleep(3)
        ids = a.sizes.split(",")
        goals = {"min_tokens_per_s": 0.1}
        prompt_tokens = None

        def predicted():
            return call("/cale/predict", {**goals, "prompt_tokens": prompt_tokens or 300, "max_tokens": a.tokens}, key)

        def run(sid):
            r = call("/v1/chat/completions", {"model": f"gemma-3n:{pct[sid]}", "messages": [{"role": "user", "content": PROMPT}],
                                              "max_tokens": a.tokens, "temperature": 0}, key)
            c, u = r["cale"], r["usage"]
            decode = (u["completion_tokens"] - 1) / max(c["seconds"] - c["first_token_s"], 1e-6)
            return {"decode_tps": decode, "first_token_s": c["first_token_s"], "prompt_tokens": u["prompt_tokens"], "tokens": u["completion_tokens"]}

        warm = run(ids[0])                                    # loads the model, and teaches a little: use a throwaway home for a pure first estimate
        prompt_tokens = warm["prompt_tokens"]
        print(f"prompt {prompt_tokens} tokens, answers of {a.tokens} tokens, load programs: {a.load}")
        print("(the model is loaded now; the predictor has learned from one answer. Restarting it for a clean first estimate.)")
        srv.terminate()
        srv.wait(20)
        wait_closed()
        shutil.rmtree(home, ignore_errors=True)
        home = tempfile.mkdtemp(prefix="cale-pred-")
        srv = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "cale_dev_server.py"), home], cwd=ROOT, env=env,
                               stdout=open(Path(home) / "server.log", "w"), stderr=subprocess.STDOUT)
        wait_ready()
        key = call("/cale/connect", {"app": "predict-check"})["api_key"]
        est = predicted()                                      # loads the model; predicts with no answer seen
        print(f"confidence before any answer: {est['confidence']}")
        rows = []
        first = run(ids[0])
        p = est["sizes"][ids[0]]
        rows.append(("first estimate", ids[0], p["tokens_per_s"], first["decode_tps"], p["first_token_s"], first["first_token_s"]))
        after = predicted()
        for sid in ids:
            actual = first if sid == ids[0] else run(sid)
            p = after["sizes"][sid]
            rows.append(("after 1st answer", sid, p["tokens_per_s"], actual["decode_tps"], p["first_token_s"], actual["first_token_s"]))
        print(f"{'when':>17} {'size':>12} | {'pred tok/s':>10} {'actual':>7} {'err':>6} | {'pred 1st-tok s':>14} {'actual':>7} {'err':>6}")
        worst = {"first estimate": 0.0, "after 1st answer": 0.0}
        for when, sid, pt, at, pf, af in rows:
            e1, e2 = (pt - at) / at, (pf - af) / af
            if not (when == "after 1st answer" and sid == ids[0]):                          # that row was the answer the numbers were just learned from
                worst[when] = max(worst[when], abs(e1))
            print(f"{when:>17} {sid:>12} | {pt:10.2f} {at:7.2f} {e1:+6.0%} | {pf:14.2f} {af:7.2f} {e2:+6.0%}")
        print(f"worst answer-speed error: first estimate {worst['first estimate']:.0%} (gate 25%), after the first answer {worst['after 1st answer']:.0%} (gate 10%)")
    finally:
        stop.set()
        for b in burners:
            b.terminate()
        srv.terminate()
        shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    main()
