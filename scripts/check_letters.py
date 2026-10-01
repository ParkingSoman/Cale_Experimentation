"""Harness check: what do models actually write first on MMLU-Pro / MMStar prompts?
If a model does not start with a bare letter, letter-logit scoring is unfair to it."""
import sys
from collections import Counter

from nest.io import load, read_jsonl
if "--nest" in sys.argv:
    from nest import elastic
    elastic.patch()
from nest.gen import generate

model, proc = load(sys.argv[1])
for task in ("mmlupro", "mmstar"):
    rows = read_jsonl(f"data/bench/{task}.jsonl")[:20]
    starts = Counter()
    for r in rows:
        out = generate(model, proc, r, max_tokens=12)
        starts[out.strip()[:12]] += 1
        print(f"{task} gold={r['answer']} -> {out!r}", flush=True)
    print(task, "first-12-char counts:", starts.most_common(8), flush=True)
