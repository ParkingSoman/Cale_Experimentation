"""Run Cale in a computer with little memory (a container with a memory limit) and check it stays inside the limit.
usage: python scripts/small_computer_check.py   (inside the container: see scripts/linux_arm64_check.sh)"""
import json, os, subprocess, sys, time, urllib.request, urllib.error
B = "http://127.0.0.1:47899"
def call(path, body=None, key=None):
    h = {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})}
    r = urllib.request.Request(B + path, data=None if body is None else json.dumps(body).encode(), headers=h)
    try:
        with urllib.request.urlopen(r, timeout=600) as x: return x.status, json.loads(x.read())
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read())
srv = subprocess.Popen([sys.executable, "scripts/cale_dev_server.py", "/tmp/home"], env=dict(os.environ, CALE_BACKEND="llama", CALE_GPU="off", CALE_LLAMA_LIB="/lib_out/libcale_llama.so", CALE_MODEL="/models/gemma-3n-E4B-it-Q4_0.cale.gguf", PYTHONPATH="."), stdout=open("/tmp/srv.log", "w"), stderr=subprocess.STDOUT)
for _ in range(60):
    try: call("/cale/status"); break
    except Exception: time.sleep(1)
key = call("/cale/connect", {"app": "small"})[1]["api_key"]
s, p = call("/cale/predict", {"min_tokens_per_s": 0.1, "prompt_tokens": 300}, key)
print("memory limit seen by Cale:", {k: v["safe"] for k, v in list(p["sizes"].items())[:1]}, "| sizes safe:", sum(1 for v in p["sizes"].values() if v["safe"]), "of", len(p["sizes"]))
print("chosen:", p["chosen"]["size"] if p.get("chosen") else None, "| needs GB:", p["chosen"]["memory_gb"] if p.get("chosen") else None)
print("smallest needs:", p["floor"]["smallest"])
s, r = call("/v1/chat/completions", {"model": "gemma-3n:smart", "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8}, key)
print("exact full size:", s, (r.get("error") or {}).get("code"), (r["choices"][0]["message"]["content"] if s == 200 else ""))
s, r = call("/v1/chat/completions", {"model": "gemma-3n:auto", "messages": [{"role": "user", "content": "Name three planets."}], "max_tokens": 30, "cale_min_tokens_per_s": 1}, key)
print("goal call:", s, r["model"] if s == 200 else r["error"]["code"], "|", (r["choices"][0]["message"]["content"][:50] if s == 200 else ""))
print("server alive:", srv.poll() is None)
srv.terminate()
