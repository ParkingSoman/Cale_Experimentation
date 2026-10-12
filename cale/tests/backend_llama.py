"""Tests for the llama.cpp backend on a real model: sizes, memory release and read-back, chat.

usage: python cale/tests/backend_llama.py MODEL.gguf [LIB]      (Linux and Windows; needs the built cale_llama library)
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def main():
    from cale.backends.llama import LlamaNested
    model, lib = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
    t = time.perf_counter()
    m = LlamaNested.load(model, lib_path=lib)
    print(f"loaded in {time.perf_counter() - t:.1f} s; {m.n_layers} layers, {len(m.levels)} sizes", flush=True)
    check("35 layers", m.n_layers == 35)
    ids = [l["id"] for l in m.levels]
    check("sizes include e4b and e2b", "e4b" in ids and "e2b" in ids, str(ids))
    full = sum(b[2] for b in m.layer_bytes)
    print(f"FFN weights {full / 1e9:.2f} GB; process memory {m.active_bytes() / 1e9:.2f} GB", flush=True)
    base = m.active_bytes() - m.resident_now()
    print(f"everything except the FFN weights: {base / 1e9:.2f} GB", flush=True)
    check("all full after load", m.all_full() and m.ffn_bytes() == full)
    from nest.levels3n import KV_BYTES_PER_TOKEN
    allowance = 2.2e9 + m.max_context * KV_BYTES_PER_TOKEN      # the context (KV cache) is part of the base memory
    check("the per-layer embedding table is not all in RAM (base under 2.2 GB plus the context)", base < allowance,
          f"{base / 1e9:.2f} GB, allowed {allowance / 1e9:.2f} GB")
    r = m.resident_now()
    check("OS says the FFN weights are in RAM", r > 0.8 * full, f"{r / 1e9:.2f} of {full / 1e9:.2f} GB")

    # memory release: down to the small size, the OS count must fall
    freed = m.reside(["e2b"])
    after = m.resident_now()
    print(f"reside e2b: accounted freed {freed / 1e6:.0f} MB; OS resident {r / 1e9:.2f} -> {after / 1e9:.2f} GB", flush=True)
    check("reside e2b frees memory", freed > 0.3 * full)
    check("the OS count fell by at least 70% of the accounted amount", (r - after) > 0.7 * freed, f"{(r - after) / 1e6:.0f} MB")
    check("missing(e4b) lists layers", len(m.missing(m.resolve("e4b"))) > 0)
    check("missing(e2b) is empty", m.missing(m.resolve("e2b")) == [])

    # a call at e2b needs nothing from disk and works
    a = m.generate("Reply with the single word: ready", 8, width="e2b")
    check("e2b answers", len(a["text"].strip()) > 0, a["text"][:40])
    check("e2b read nothing from disk", a["stats"]["disk_mb"] == 0)

    # a call at e4b reads the missing weights, then gives them back
    b = m.generate("Reply with the single word: ready", 8, width="e4b")
    check("e4b answers", len(b["text"].strip()) > 0, b["text"][:40])
    check("e4b read weights from disk first", b["stats"]["disk_mb"] > 100, str(b["stats"]["disk_mb"]))
    check("memory went back to the e2b state after the call", m.state == m.resident and m.resident_now() < r * 0.8,
          f"{m.resident_now() / 1e9:.2f} GB")

    # speed of switching and basic stats
    ms = [m.set(w) for w in ("e4b", "e2b", "e4b", "e2b")]
    print("switch ms:", [round(x, 2) for x in ms], flush=True)
    check("switching is fast (under 50 ms)", max(ms) < 50)
    st = a["stats"]
    check("stats are filled", st["tokens"] > 0 and st["prompt_tokens"] > 0 and st["seconds"] > 0, str(st))

    # streaming pieces join to the full text, and a non-ASCII answer is not cut in half
    parts = [ev["delta"] for ev in m.chat_stream([{"role": "user", "content": "Say hello in Japanese and in Hindi."}], 40, width="e2b") if "delta" in ev]
    text = "".join(parts)
    check("stream gives pieces", len(parts) > 3)
    check("no broken characters", "�" not in text, text[:60])

    # holds: e2b held, then the full model again
    m.reside(["e4b"])
    check("reside e4b reads everything back", m.all_full() and m.resident_now() > 0.8 * full)
    m.close()
    print("\nALL PASS" if not FAILS else f"\nFAILED: {FAILS}", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
