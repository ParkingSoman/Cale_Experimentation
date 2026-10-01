"""Download fixed, seeded subsets of the benchmarks into data/bench/*.jsonl."""
import json
import os
import random

from datasets import load_dataset

OUT = "data/bench"
os.makedirs(f"{OUT}/img", exist_ok=True)
rng = random.Random(1234)
LETTERS = "ABCDEFGHIJ"


def write(name, rows):
    with open(f"{OUT}/{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows), flush=True)


def sample(ds, n):
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    return [ds[i] for i in idx[:n]]


# Knowledge: MMLU-Pro, letter choice (A-J), scored by next-token log-probs.
ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
rows = []
for r in sample(ds, 200):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(r["options"]))
    rows.append({"id": f"mmlupro-{r['question_id']}", "category": r["category"],
                 "user": f"{r['question']}\n\n{opts}\n\nAnswer with only the letter of the correct option.",
                 "choices": LETTERS[: len(r["options"])], "answer": r["answer"], "image": None})
write("mmlupro", rows)

# Math: GSM8K, generated answer, final number compared.
ds = load_dataset("openai/gsm8k", "main", split="test")
rows = []
for i, r in enumerate(sample(ds, 100)):
    rows.append({"id": f"gsm8k-{i}", "user": r["question"] + "\n\nThink step by step. End with 'The answer is N.'",
                 "answer": r["answer"].split("####")[-1].strip().replace(",", ""), "image": None})
write("gsm8k", rows)

# Instruction following: IFEval, strict prompt-level.
ds = load_dataset("google/IFEval", split="train")
rows = []
for r in sample(ds, 60):
    rows.append({"id": f"ifeval-{r['key']}", "user": r["prompt"], "key": r["key"],
                 "instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"], "image": None})
write("ifeval", rows)

# Vision, perception + reasoning: MMStar, letter choice (A-D).
ds = load_dataset("Lin-Chen/MMStar", split="val")
rows = []
for r in sample(ds, 100):
    path = f"{OUT}/img/mmstar-{r['index']}.png"
    r["image"].convert("RGB").save(path)
    rows.append({"id": f"mmstar-{r['index']}", "category": r["category"],
                 "user": r["question"] + "\n\nAnswer with only the letter of the correct option.",
                 "choices": "ABCD", "answer": r["answer"], "image": path})
write("mmstar", rows)

# Vision, reading text: OCRBench, short generated answer, containment match.
ds = load_dataset("echo840/OCRBench", split="test")
rows = []
for i, r in enumerate(sample(ds, 100)):
    path = f"{OUT}/img/ocrbench-{i}.png"
    r["image"].convert("RGB").save(path)
    ans = r["answer"] if isinstance(r["answer"], list) else [r["answer"]]
    rows.append({"id": f"ocrbench-{i}", "category": r["question_type"], "user": r["question"],
                 "answer": ans, "image": path})
write("ocrbench", rows)
