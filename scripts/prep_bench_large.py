"""Larger seeded subsets for final comparisons: MMLU-Pro 1000, MMStar 300."""
import json, os, random
from datasets import load_dataset
OUT = "data/bench"; rng = random.Random(4321); LETTERS = "ABCDEFGHIJ"
def sample(ds, n):
    idx = list(range(len(ds))); rng.shuffle(idx); return [ds[i] for i in idx[:n]]
def write(name, rows):
    with open(f"{OUT}/{name}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows), flush=True)
ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
rows = []
for r in sample(ds, 1000):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(r["options"]))
    rows.append({"id": f"mmlupro-{r['question_id']}", "category": r["category"],
                 "user": f"{r['question']}\n\n{opts}\n\nAnswer with only the letter of the correct option.",
                 "choices": LETTERS[: len(r["options"])], "answer": r["answer"], "image": None})
write("mmlupro1k", rows)
ds = load_dataset("Lin-Chen/MMStar", split="val")
rows = []
for r in sample(ds, 300):
    path = f"{OUT}/img/mmstar-{r['index']}.png"
    if not os.path.exists(path): r["image"].convert("RGB").save(path)
    rows.append({"id": f"mmstar-{r['index']}", "category": r["category"],
                 "user": r["question"] + "\n\nAnswer with only the letter of the correct option.",
                 "choices": "ABCD", "answer": r["answer"], "image": path})
write("mmstar300", rows)
