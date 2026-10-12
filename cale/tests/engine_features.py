"""Engine checks that must hold on every engine (MLX on Mac, llama.cpp on Windows and Linux). Loads the model.

usage: python cale/tests/engine_features.py [MODEL]      MODEL: a folder (Mac) or a .gguf file. Default: the installed Cale's model.
Environment: CALE_BACKEND, CALE_LLAMA_LIB (as for the server).
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cale import backends, paths  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAILS.append(name)


def ask(m, text, max_tokens=60, **kw):
    out, done = "", None
    for ev in m.chat_stream([{"role": "user", "content": text}], max_tokens=max_tokens, **kw):
        out += ev.get("delta", "")
        done = ev.get("done", done)
    return out, done


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else str(paths.MODEL)
    m = backends.load_model(path)
    sup = getattr(m, "supports", {})
    print(f"engine={backends.name()} max_context={getattr(m, 'max_context', None)} supports={sup}", flush=True)
    check("engine reports a max context", getattr(m, "max_context", 0) >= 4096)

    base, _ = ask(m, "Count from 1 to 12, separated by commas, nothing else.", 60)
    check("greedy answer is not empty", len(base) > 10, base)

    if sup.get("stop"):
        out, done = ask(m, "Count from 1 to 12, separated by commas, nothing else.", 60, stop=["6"])
        check("stop: text ends before the stop sequence", "6" not in out and out.strip().endswith("5,") or ("6" not in out and "5" in out), out)
        check("stop: finish reason is stop", done["finish"] == "stop", done)
        out2, _ = ask(m, "Count from 1 to 12, separated by commas, nothing else.", 60, stop=["zzzz-never"])
        check("stop: a sequence that never appears changes nothing", out2 == base, (out2, base))

    if sup.get("seed"):
        a, _ = ask(m, "Write one short sentence about the sea.", 30, temperature=1.5, seed=7)
        b, _ = ask(m, "Write one short sentence about the sea.", 30, temperature=1.5, seed=7)
        c, _ = ask(m, "Write one short sentence about the sea.", 30, temperature=1.5, seed=8)
        check("seed: same seed repeats", a == b, (a, b))
        check("seed: different seed differs", a != c, (a, c))
        u1, _ = ask(m, "Write one short sentence about the sea.", 30, temperature=1.5)
        u2, _ = ask(m, "Write one short sentence about the sea.", 30, temperature=1.5)
        check("seed: no seed gives different answers", u1 != u2, (u1, u2))

    if sup.get("top_p"):
        g, _ = ask(m, "Name a colour.", 8)
        t, _ = ask(m, "Name a colour.", 8, temperature=1.0, top_p=0.0001, seed=3)
        check("top_p: a tiny top_p acts like greedy", g == t, (g, t))
    else:
        print("SKIP top_p (this engine does not support it yet)", flush=True)

    # A long prompt must neither crash nor change the answer. On MLX this used to allocate the whole attention table at once.
    filler = " ".join(f"Item {i} is a number that appears in the list." for i in range(900))      # about 9k tokens
    long_prompt = filler + "\n\nQuestion: what is 2+2? Answer with one word."
    t0 = time.time()
    out, done = ask(m, long_prompt, 6)
    check("long prompt (about 9k tokens) answers", done is not None and done["prompt_tokens"] > 7000, done)
    print(f"  prompt_tokens={done['prompt_tokens']} in {time.time() - t0:.1f}s answer={out!r}", flush=True)

    if backends.name() == "mlx":
        import nest.gemma3n as g3
        g3.PREFILL_STEP, old = 256, g3.PREFILL_STEP
        out_small, _ = ask(m, long_prompt, 6)
        g3.PREFILL_STEP = old
        check("chunked reading gives the same answer as large pieces", out_small == out, (out_small, out))

    print("\nALL PASS" if not FAILS else f"\nFAILED: {FAILS}", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
