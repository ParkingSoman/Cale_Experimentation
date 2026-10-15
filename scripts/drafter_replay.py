#!/usr/bin/env python3
"""Offline replay: how much of a saved answer would a copy-drafter have guessed? No model, no engine, numbers only.

usage: python3 scripts/drafter_replay.py --tokenizer models/gemma-3n-e4b-4bit/tokenizer.json --out results/drafter-replay.json NAME=FILE [NAME=FILE ...]
FILE is a text file, or a JSON file with a "text" field (optionally "prompt"). Each file is one saved answer.

Two copy-drafters, both need no training:
  answer-copy: look for the last N tokens earlier in the ANSWER, and guess the tokens that followed there.
  prompt-copy: the same, but also look in the PROMPT text (when the prompt is an image there is no text: it adds nothing).
Verifier model: one pass checks up to L guessed tokens and always yields one token of its own, so
  tokens per pass = tokens / passes. Plain decoding is 1.0. This ignores that a pass with more tokens may cost more
  (on an arithmetic-limited CPU it can cost nearly as much per token), so it is an UPPER bound on the gain.
"""
import argparse
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer


def replay(prompt_ids, out_ids, use_prompt, n_match=3, max_draft=8):
    hist = list(prompt_ids) if use_prompt else []
    passes = accepted = drafted = 0
    i = 0
    while i < len(out_ids):
        guess = []
        for n in range(n_match, 1, -1):                         # longest match first
            if len(hist) >= n:
                key = hist[-n:]
                for j in range(len(hist) - n - 1, -1, -1):      # most recent earlier occurrence
                    if hist[j:j + n] == key:
                        guess = hist[j + n:j + n + max_draft]
                        break
            if guess:
                break
        k = 0
        while k < len(guess) and i + k < len(out_ids) and guess[k] == out_ids[i + k]:
            k += 1
        drafted += len(guess)
        accepted += k
        step = min(k + 1, len(out_ids) - i)                      # the verified tokens plus the pass's own token
        hist.extend(out_ids[i:i + step])
        i += step
        passes += 1
    return {"tokens": len(out_ids), "passes": passes, "tokens_per_pass": round(len(out_ids) / passes, 3),
            "drafted": drafted, "accepted": accepted, "accept_rate": round(accepted / drafted, 3) if drafted else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("items", nargs="+")
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    result = {}
    for item in a.items:
        name, _, path = item.partition("=")
        raw = Path(path).read_text()
        text, prompt = raw, ""
        if path.endswith(".json"):
            d = json.loads(raw)
            text, prompt = d.get("text", ""), d.get("prompt", "")
        out_ids = tok.encode(text, add_special_tokens=False).ids
        pr_ids = tok.encode(prompt, add_special_tokens=False).ids if prompt else []
        result[name] = {"answer_copy": replay([], out_ids, False), "prompt_copy": replay(pr_ids, out_ids, True)}
        r = result[name]
        print(f"{name:20s} {r['answer_copy']['tokens']:5d} tokens | answer-copy {r['answer_copy']['tokens_per_pass']:.2f} tok/pass "
              f"(accept {r['answer_copy']['accept_rate']}) | prompt-copy {r['prompt_copy']['tokens_per_pass']:.2f} tok/pass", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    sys.exit(main())
