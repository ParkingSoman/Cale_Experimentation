"""Build the calibration set and the held-out drift set.

Each record is one chat-style sample: {"id", "source", "user", "assistant", "image"}.
"image" is a path to a PNG file, or null. Calibration and held-out samples come
from disjoint slices of each source, so the drift check never sees calibration data.

Output:
  data/calib.jsonl     - used to measure neuron and layer importance
  data/heldout.jsonl   - used only to measure drift from the full model
  data/images/*.png
"""
import itertools
import json
import os
import random

from datasets import load_dataset

random.seed(0)
OUT = "data"
os.makedirs(f"{OUT}/images", exist_ok=True)

N_CALIB = 32   # per source
N_HELD = 25    # per source
MAX_CHARS = 2400


def clip(s):
    return s[:MAX_CHARS]


def take(it, n):
    return list(itertools.islice(it, n))


def web():
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    for r in ds:
        if len(r["text"]) > 1500:
            yield {"user": "Summarize the following text.\n\n" + clip(r["text"]), "assistant": None}


def code():
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True)
    for r in ds:
        yield {"user": clip(r["instruction"] + ("\n" + r["input"] if r["input"] else "")), "assistant": clip(r["output"])}


def math():
    ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    for r in ds:
        yield {"user": r["question"], "assistant": r["answer"]}


def multilingual():
    langs = ["fr", "es", "de", "zh", "ja", "hi", "ar", "ru"]
    its = [iter(load_dataset("wikimedia/wikipedia", f"20231101.{l}", split="train", streaming=True)) for l in langs]
    while True:
        for it in its:
            r = next(it)
            while len(r["text"]) < 800:
                r = next(it)
            yield {"user": "Explain the main points of this article.\n\n" + clip(r["text"]), "assistant": None}


def chat():
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    for r in ds:
        m = r["messages"]
        if len(m) >= 2 and m[0]["role"] == "user":
            yield {"user": clip(m[0]["content"]), "assistant": clip(m[1]["content"])}


def images(subset):
    def gen():
        ds = load_dataset("HuggingFaceM4/the_cauldron", subset, split="train", streaming=True)
        for r in ds:
            if not r["images"] or not r["texts"]:
                continue
            t = r["texts"][0]
            yield {"user": t["user"], "assistant": t["assistant"], "_img": r["images"][0]}
    return gen


SOURCES = {
    "web": web,
    "code": code,
    "math": math,
    "multilingual": multilingual,
    "chat": chat,
    "img_chart": images("chartqa"),
    "img_doc": images("docvqa"),
    "img_diagram": images("ai2d"),
    "img_photo": images("vqav2"),
    "img_text": images("textcaps"),
}


def main():
    calib, held = [], []
    for name, gen in SOURCES.items():
        rows = take(gen(), (N_CALIB + N_HELD) * 3)
        random.shuffle(rows)
        rows = rows[: N_CALIB + N_HELD]
        for i, r in enumerate(rows):
            rid = f"{name}-{i}"
            img = r.pop("_img", None)
            if img is not None:
                path = f"{OUT}/images/{rid}.png"
                img.convert("RGB").save(path)
                r["image"] = path
            else:
                r["image"] = None
            r.update(id=rid, source=name)
            (calib if i < N_CALIB else held).append(r)
        print(name, len(rows), flush=True)
    for fn, rows in (("calib.jsonl", calib), ("heldout.jsonl", held)):
        with open(f"{OUT}/{fn}", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("calib", len(calib), "heldout", len(held))


if __name__ == "__main__":
    main()
