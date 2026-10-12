"""Gate C: nest/gemma3n.py at full size gives the same tokens as plain mlx-vlm E4B.

Run: .venv/bin/python scripts/g3n_gate_c.py   (writes results/g3n-gate-c.json)
"""
import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODEL = ROOT / "models" / "gemma-3n-e4b-4bit"
PROMPTS = [
    "Say hello in five words.",
    "What is the capital of Australia, and why is it not Sydney?",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain in two sentences how a refrigerator works.",
    "Give three tips for a first-time marathon runner.",
    "Translate to French: The meeting is moved to Thursday afternoon.",
    "What is 17 times 23? Show the steps.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
]
N = 48


def plain():
    """Greedy text from the untouched model, with the same prompt tokens and loop as chat_stream."""
    from mlx_vlm import load
    from mlx_vlm.models.gemma3n.language import logit_softcap
    model, proc = load(str(MODEL))
    lm, tok = model.language_model, proc.tokenizer
    outs = []
    for p in PROMPTS:
        text = proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": p}]}],
                                        tokenize=False, add_generation_prompt=True)
        cache, ids, out = lm.make_cache(), mx.array(tok.encode(text, add_special_tokens=False))[None], []
        for _ in range(N):
            h = lm.model(ids, cache=cache)
            lg = logit_softcap(lm.final_logit_softcapping, lm.model.embed_tokens.as_linear(h[:, -1:]))
            ids = mx.argmax(lg, axis=-1)
            v = int(ids.item())
            if v in (1, 106):
                break
            out.append(v)
        outs.append(tok.decode(out, skip_special_tokens=True))
    return outs, (model, proc)


def main():
    ref, loaded = plain()
    from nest.gemma3n import NestedGemma3n
    t = time.perf_counter()
    m = NestedGemma3n.load(MODEL, width="e4b", loaded=loaded)
    load_s = time.perf_counter() - t
    ours = [m.generate(p, N)["text"] for p in PROMPTS]
    same = sum(a.strip() == b.strip() for a, b in zip(ref, ours))
    # match length in characters, to see how far the texts agree before they split
    pref = [next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))) / max(len(a), 1)
            for a, b in zip(ref, ours)]
    print(f"gate C: {same}/{len(PROMPTS)} identical; mean agreeing prefix {sum(pref) / len(pref):.2f}")
    for p, a, b in zip(PROMPTS, ref, ours):
        if a.strip() != b.strip():
            print("  DIFF", p[:40], "\n   ref :", repr(a[:100]), "\n   ours:", repr(b[:100]))
    sizes = []
    for lv in m.levels:
        sw = m.set(lv["id"])
        r = m.generate(PROMPTS[1], N)
        s = r["stats"]
        sizes.append({"id": lv["id"], "pct": lv["pct"], "switch_ms": round(sw, 2), "tok_per_s":
                      round(s["tokens"] / s["seconds"], 1), "sample": r["text"][:80]})
        print(f'  {lv["id"]:16} {lv["pct"]:3}%  switch {sw:5.2f} ms  {sizes[-1]["tok_per_s"]:5.1f} tok/s  {r["text"][:60]!r}')
    out = {"identical": same, "total": len(PROMPTS), "mean_prefix": sum(pref) / len(pref), "load_s": round(load_s, 1),
           "active_gb": round(mx.get_active_memory() / 1e9, 2), "sizes": sizes}
    (ROOT / "results" / "g3n-gate-c.json").write_text(json.dumps(out, indent=1))
    print("active GB", out["active_gb"], "load s", out["load_s"])


if __name__ == "__main__":
    main()
