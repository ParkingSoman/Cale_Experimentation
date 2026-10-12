"""Speed of Cale's own engine on this machine: prompt speed, answer speed, memory and load time.

usage: xplat_bench.py MODEL OUT.json [LIB]      MODEL: a folder (Mac, MLX) or a .gguf file (Windows, Linux)
Runs the full size and the smallest size of the one loaded model, with whichever engine this computer uses.
Prompt speed: a prompt of about 1000 tokens, one new token. Answer speed: the extra time for 64 more tokens.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FILLER = ("The quick brown fox jumps over the lazy dog while the committee reviews the quarterly report on "
          "water use, road repairs and library hours. ")
ASK = "\n\nIn one sentence, what is this text about?"


def timed(m, prompt, n):
    best = None
    for _ in range(1):
        s = m.generate(prompt, n)["stats"]
        best = s if best is None or s["seconds"] < best["seconds"] else best
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("lib", nargs="?")
    a = ap.parse_args()
    from cale import backends, oslayer
    t0 = time.perf_counter()
    if backends.name() == "mlx":
        m = backends.load_model(a.model)
    else:
        from cale.backends.llama import LlamaNested
        m = LlamaNested.load(a.model, lib_path=a.lib)
    load_s = time.perf_counter() - t0
    long_prompt = FILLER * 20 + ASK
    m.generate("Say hi.", 4)                                   # warm up
    out = {"os": platform.system(), "machine": platform.machine(), "backend": backends.name(),
           "repacked": getattr(getattr(m, "lib", None), "repacked", None), "load_s": round(load_s, 1),
           "active_gb_after_load": round(m.active_bytes() / 1e9, 2), "sizes": {}}
    lv_full = max(m.levels, key=lambda l: l["pct"])
    lv_small = min(m.levels, key=lambda l: l["pct"])
    for name, lv in (("full", lv_full), ("small", lv_small)):
        m.set(lv["id"])
        s1 = timed(m, long_prompt, 1)
        s65 = timed(m, long_prompt, 65)
        decode_s = max(s65["seconds"] - s1["seconds"], 1e-6)
        out["sizes"][name] = {
            "id": lv["id"], "pct": lv["pct"], "prompt_tokens": s1["prompt_tokens"],
            "prefill_tok_s": round(s1["prompt_tokens"] / s1["seconds"], 1),
            "decode_tok_s": round((s65["tokens"] - s1["tokens"]) / decode_s, 2),
            "memory_gb": round(m.active_bytes() / 1e9, 2)}
        print(name, out["sizes"][name], flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
