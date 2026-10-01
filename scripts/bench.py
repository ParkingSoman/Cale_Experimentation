"""Run the benchmark subsets on one model and one or more levels.

usage: bench.py MODEL_DIR NAME [--nest] [--level LEVELS_JSON] [--tasks mmlupro,gsm8k,...] [--limit N]
Writes results/bench/NAME.json (per level, per task: accuracy and per-item outputs).
"""
import argparse
import json
import os
import re
import time

import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("name")
ap.add_argument("--nest", action="store_true")
ap.add_argument("--level")
ap.add_argument("--only", help="comma list of level names to run")
ap.add_argument("--tasks", default="mmlupro,mmstar,ocrbench,gsm8k,ifeval")
ap.add_argument("--limit", type=int)
args = ap.parse_args()

if args.nest:
    from nest import elastic
    elastic.patch()
from nest.gen import choose_letter, generate
from nest.io import load, read_jsonl


def num(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$", ""))
    return m[-1].replace(",", "").rstrip(".") if m else None


def run_task(task, model, proc, letter_ids):
    rows = read_jsonl(f"data/bench/{task}.jsonl")[: args.limit]
    items = []
    for r in rows:
        if task in ("mmlupro", "mmstar"):
            pred = choose_letter(model, proc, r, letter_ids)
            ok = pred == r["answer"]
        elif task == "gsm8k":
            pred = generate(model, proc, r, max_tokens=320)
            m = re.search(r"answer is\s*\$?(-?[\d,]*\.?\d+)", pred)
            got = m.group(1).replace(",", "") if m else num(pred)
            ok = got is not None and abs(float(got) - float(r["answer"])) < 1e-6 if got not in (None, "", "-") else False
        elif task == "ocrbench":
            pred = generate(model, proc, r, max_tokens=48)
            ok = any(str(a).strip().lower() in pred.strip().lower() for a in r["answer"])
        elif task == "ifeval":
            from lm_eval.tasks.ifeval.utils import InputExample, test_instruction_following_strict
            pred = generate(model, proc, r, max_tokens=400)
            inp = InputExample(key=r["key"], instruction_id_list=r["instruction_id_list"],
                               prompt=r["user"], kwargs=r["kwargs"])
            ok = test_instruction_following_strict(inp, pred).follow_all_instructions
        items.append({"id": r["id"], "pred": pred, "ok": bool(ok)})
        mx.clear_cache()
    return {"acc": sum(i["ok"] for i in items) / len(items), "n": len(items), "items": items}


mx.set_cache_limit(1024**3)
model, proc = load(args.model)
tk = proc.tokenizer
letter_ids = {c: tk.encode(" " + c, add_special_tokens=False)[0] for c in "ABCDEFGHIJ"}
assert all(len(tk.encode(" " + c, add_special_tokens=False)) == 1 for c in "ABCDEFGHIJ")
levels = json.load(open(args.level)) if args.level else {"full": None}
if args.only:
    levels = {k: v for k, v in levels.items() if k in args.only.split(",")}
os.makedirs("results/bench", exist_ok=True)
path = f"results/bench/{args.name}.json"
out = json.load(open(path)) if os.path.exists(path) else {}
for lname, lv in levels.items():
    if args.nest:
        if lv is None:
            elastic.set_level(model)
        else:
            elastic.set_level(model, lv.get("widths"), lv.get("skip", ()))
    out.setdefault(lname, {})
    for task in args.tasks.split(","):
        t = time.time()
        res = run_task(task, model, proc, letter_ids)
        res["seconds"] = time.time() - t
        out[lname][task] = res
        print(f"[{args.name}/{lname}] {task}: {res['acc']:.3f} (n={res['n']}, {res['seconds']:.0f}s)", flush=True)
        json.dump(out, open(path, "w"), indent=1)
print("peak GB", mx.get_peak_memory() / 1e9)
