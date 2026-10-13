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
        us = [ask(m, "Write one short sentence about the sea.", 30, temperature=1.5)[0] for _ in range(4)]      # a peaked prompt can repeat once by chance
        check("seed: no seed gives different answers", len(set(us)) >= 2, us)

    if sup.get("top_p"):
        g, _ = ask(m, "Name a colour.", 8)
        t, _ = ask(m, "Name a colour.", 8, temperature=1.0, top_p=0.0001, seed=3)
        check("top_p: a tiny top_p acts like greedy", g == t, (g, t))
    else:
        print("SKIP top_p (this engine does not support it yet)", flush=True)

    # A long prompt must neither crash nor change the answer. On MLX a 45k-token prompt once crashed the engine (a manual check: scripts/long_prompt.py).
    items = 150                                                           # about 14 tokens per item: a prompt of roughly 2k tokens (quick even on a slow computer)
    filler = " ".join(f"Item {i} is a number that appears in the list." for i in range(items))
    long_prompt = filler + "\n\nQuestion: what is 2+2? Answer with one word."
    t0 = time.time()
    out, done = ask(m, long_prompt, 6)
    check("a long prompt (about 2k tokens) answers", done is not None and done["prompt_tokens"] > 1500, done)
    print(f"  prompt_tokens={done['prompt_tokens']} in {time.time() - t0:.1f}s answer={out!r}", flush=True)

    if backends.name() == "mlx":
        import nest.gemma3n as g3
        g3.PREFILL_STEP, old = 256, g3.PREFILL_STEP
        out_small, _ = ask(m, long_prompt, 6)
        g3.PREFILL_STEP = old
        # Smaller pieces change rounding a little, so a near-tie word may differ ("Four" and "Forty"). The start must agree.
        check("chunked reading gives (nearly) the same answer as large pieces", out_small[:2] == out[:2], (out_small, out))
        # A fact in the middle of a prompt longer than the model's local window (512) must be found. Reading in pieces longer than
        # that window once made the model quote another code from the same document.
        import random
        r = random.Random(5)
        adj, noun = "quiet old narrow bright heavy slow green cold wide small hidden broken".split(), "harbor lantern ledger bridge orchard tower market garden engine archive window river".split()
        verb, place = "watched crossed guarded recorded followed measured hid carried repaired".split(), "Veln Marrow Ostia Quill Dunmere Harrowgate Tessel Brannoch".split()
        adv = "carefully slowly at dawn every year without complaint during the winter".split()
        sent = lambda: f"The {r.choice(adj)} {r.choice(noun)} of {r.choice(place)} {r.choice(verb)} the {r.choice(adj)} {r.choice(noun)} {r.choice(adv)}."
        paras = [" ".join(sent() for _ in range(8)) for _ in range(54)]
        paras[10] += " The access code for the red vault is 5520-ANCHOR."
        paras[27] += " The access code for the blue vault is 7391-LANTERN."
        paras[44] += " The access code for the green vault is 8814-MARBLE."
        needle = "DOCUMENT:\n" + "\n\n".join(paras) + "\n\nQUESTION: What is the access code for the blue vault? Reply with the code only."
        out_n, done_n = ask(m, needle, 20)
        check("a fact in the middle of a long prompt is found", "7391-LANTERN" in out_n and done_n["prompt_tokens"] > 4000, (out_n, done_n["prompt_tokens"]))

    print("\nALL PASS" if not FAILS else f"\nFAILED: {FAILS}", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
