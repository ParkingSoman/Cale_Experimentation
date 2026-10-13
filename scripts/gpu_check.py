"""Does a GPU give the same answers as the CPU, at several Cale sizes, and how fast is each setup?

usage: python scripts/gpu_check.py MODEL.gguf [--lib PATH] [--sizes e4b,e2b,...] [--tokens 48]

For each setup (CPU only, then each GPU found) and each size it measures: NLL of a fixed text (must match the CPU within
0.02 nats), the first generated tokens, prompt speed and answer speed. Standard library only.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cale.backends import llamalib  # noqa: E402
from nest.levels3n import BLOCK, load_levels  # noqa: E402

TEXT = ("The water cycle moves water between the oceans, the air and the land. Heat from the sun makes water rise as vapor. "
        "In the cold air the vapor turns back into drops, which fall as rain or snow. Rivers carry the water back to the sea. ") * 3
PROMPT = "Explain in three sentences why the sky is blue. " + "Use plain words. " * 40


def widths_for(level, n_layers):
    full = all(level["blocks"][i] == 2 for i in range(n_layers)) and not level["skip"]
    return None if full else [0 if i in level["skip"] else level["blocks"][i] * BLOCK for i in range(n_layers)]


def say(text):
    print(text, flush=True)


def run(model, lib_path, gpu, sizes, n_tokens, n_gpu_layers):
    say(f"  loading (gpu={gpu}, layers={n_gpu_layers})")
    lib = llamalib.Lib(model, n_threads=4, n_ctx=4096, lib_path=lib_path, gpu=gpu, n_gpu_layers=n_gpu_layers)
    rows = []
    try:
        levels = {l["id"]: l for l in load_levels(lib.n_layers)}
        say(f"  loaded: {lib.placement}")
        text_ids = lib.tokenize(TEXT)
        prompt_ids = lib.tokenize(lib.chat_text([{"role": "user", "content": PROMPT}]), add_special=False)
        for sid in sizes:
            lib.set_widths(widths_for(levels[sid], lib.n_layers))
            say(f"    {sid}: nll")
            nll = lib.nll(text_ids, first=8)
            say(f"    {sid}: prompt of {len(prompt_ids)} tokens")
            t0 = time.perf_counter()
            lib.begin(prompt_ids, temperature=0.0)
            toks = [lib.next()]                     # a GPU finishes the prompt only when the first token is read
            t1 = time.perf_counter()
            for _ in range(n_tokens - 1):
                t = lib.next()
                if t < 0 or lib.is_eog(t):
                    break
                toks.append(t)
            t2 = time.perf_counter()
            text = b"".join(lib.piece(t) for t in toks).decode("utf-8", "replace")
            rows.append({"size": sid, "nll": nll, "prompt_tps": len(prompt_ids) / (t1 - t0),
                         "decode_tps": (len(toks) - 1) / max(t2 - t1, 1e-9), "tokens": toks, "text": text})
        return lib.placement, rows
    finally:
        lib.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lib")
    ap.add_argument("--sizes", default="e4b,e2b")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--gpu-layers", type=int, default=-1, help="-1 = llama.cpp's fit")
    ap.add_argument("--only", help="run only the device whose name contains this text (and the CPU baseline)")
    ap.add_argument("--no-cpu", action="store_true", help="skip the CPU baseline (no comparison)")
    a = ap.parse_args()
    sizes = a.sizes.split(",")
    devs = llamalib.devices(a.lib)
    print("devices:", *[f"{d['handle']}:{d['type']}:{d['name']} free={d['free'] >> 20}MB" for d in devs] or ["none"])
    base_place, base = ("", []) if a.no_cpu else run(a.model, a.lib, None, sizes, a.tokens, 0)
    print(base_place)
    for r in base:
        print(f"  CPU  {r['size']:>5} nll={r['nll']:.4f} prompt={r['prompt_tps']:.1f} decode={r['decode_tps']:.1f} | {r['text'][:70]!r}")
    for d in devs:
        if a.only and a.only.lower() not in (d["name"] + d["description"]).lower():
            continue
        try:
            place, rows = run(a.model, a.lib, d["handle"], sizes, a.tokens, a.gpu_layers)
        except Exception as e:
            print(f"{d['name']}: FAILED to run: {e}")
            continue
        print(place)
        for i, r in enumerate(rows):
            b = base[i] if i < len(base) else None
            if b:
                same = sum(1 for x, y in zip(r["tokens"], b["tokens"]) if x == y)
                ok = "OK " if abs(r["nll"] - b["nll"]) <= 0.02 else "BAD"
                cmp = f"(cpu {b['nll']:.4f}) {ok}"
            else:
                same, cmp = 0, ""
            print(f"  GPU  {r['size']:>5} nll={r['nll']:.4f} {cmp} prompt={r['prompt_tps']:.1f} "
                  f"decode={r['decode_tps']:.1f} same-first-tokens={same} | {r['text'][:70]!r}")


if __name__ == "__main__":
    main()
