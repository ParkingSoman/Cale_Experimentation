"""Tests for reading prompts faster: a step with no output token skips the layers without their own memory (llama/patches/0003),
and Cale reads the last prompt token in a step of its own (cale_set_option key 4).

usage: python cale/tests/prefill_exit.py MODEL.gguf [LIB]      (llama engine: Linux, Windows; on the Mac engine (MLX) only the rule is checked)

What is checked:
  1. the rule that decides when the last token is read alone (pure Python)
  2. the early exit changes no number: the same prompts with CALE_NO_KV_EXIT=1 and without it give the same scores, bit for bit
  3. reading the last token alone makes prompts faster (alternating runs in one process)
  4. the backend applies the rule when a size is set
"""
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FAILS = []
BASE = ("The harbour town of Wexford keeps its records in a stone cellar. In 1814 the clerk Ansel Brightwater wrote that the tide "
        "mills ground 41 sacks of barley on Monday, 38 on Tuesday and 52 on Wednesday, and that the miller's daughter Ione counted "
        "every sack twice. ")
ASK = "\nHow many sacks were ground on Wednesday?"


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def child(model, lib, out):
    """Score three prompts at three sizes: the most likely next tokens and 12 greedy tokens. Written to `out` as JSON."""
    from cale.backends.llama import LlamaNested
    m = LlamaNested.load(model, lib_path=lib)
    prompts = {"short": "Name three colours of the rainbow.", "mid": BASE * 3 + ASK}
    levels = [max(m.levels, key=lambda l: l["pct"]), min(m.levels, key=lambda l: abs(l["pct"] - 62)), min(m.levels, key=lambda l: l["pct"])]
    res = {}
    for lv in levels:
        m.set(lv["id"])
        for name, text in prompts.items():
            ids = m.lib.tokenize(text)
            m.lib.begin(ids, 0.0)
            top = m.lib.topk(5)
            toks = []
            for _ in range(12):
                t = m.lib.next()
                if t < 0 or m.lib.is_eog(t):
                    break
                toks.append(t)
            res[f"{lv['id']}/{name}"] = {"top5": [[i, p] for i, p in top], "greedy": toks}
    Path(out).write_text(json.dumps(res))


def main():
    os.environ.setdefault("CALE_SLOTS", "3")                # answers that run together: the slot test below needs more than one
    from cale import backends
    from cale import gemma3n_facts as facts
    full = facts.full_level()

    # 1. the rule
    f = facts.shared_share(full)
    check("layers 20 and up are about 43% of the work of the full size", 0.35 < f < 0.5, f"{f:.3f}")
    check("a smaller size that skips some of those layers has a smaller share", facts.shared_share(min(facts.levels(), key=lambda l: l["eff_params_b"])) < f)
    slow = facts.hold_last_min(full, 27.6, 7.0)       # a CPU laptop: a step on its own is cheap next to reading a prompt
    fast = facts.hold_last_min(full, 3000.0, 100.0)   # a fast computer: a step on its own costs the time of many prompt tokens
    check("the last token alone pays from a few tokens on a slow computer", 2 <= slow <= 8, str(slow))
    check("it needs a longer last piece on a fast computer", fast > slow, f"{fast} against {slow}")
    check("no speeds, no gain: the rule says never", facts.hold_last_min(full, 0.0, 7.0) == 0)

    if backends.name() != "llama":
        print("the early exit is part of the llama engine: the rest is skipped on this computer (the Mac engine skips unused work by itself)")
        return 1 if FAILS else 0
    model, lib = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)

    # 2. the early exit changes no number
    env = dict(os.environ, CALE_HOLD_LAST="3", PYTHONPATH=str(ROOT))
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        outs = {}
        for tag, extra in (("on", {}), ("off", {"CALE_NO_KV_EXIT": "1"})):
            outs[tag] = str(Path(d) / f"{tag}.json")
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", model, lib or "", outs[tag]],
                               cwd=ROOT, env={**env, **extra}, capture_output=True, text=True, timeout=1500)
            check(f"scoring run {tag} finished", r.returncode == 0 and Path(outs[tag]).exists(), (r.stderr or "")[-200:])
        if all(Path(p).exists() for p in outs.values()):
            a, b = json.loads(Path(outs["on"]).read_text()), json.loads(Path(outs["off"]).read_text())
            same = all(a[k]["top5"] == b[k]["top5"] and a[k]["greedy"] == b[k]["greedy"] for k in a)
            check("the early exit changes no number (scores and answers equal, bit for bit, at 3 sizes)", same,
                  "" if same else str({k: (a[k]["top5"][0], b[k]["top5"][0]) for k in a if a[k]["top5"] != b[k]["top5"]}))

    # 3. speed, alternating in one process
    from cale.backends.llama import LlamaNested
    m = LlamaNested.load(model, lib_path=lib)
    if not hasattr(m.lib.c, "cale_set_option"):
        check("this library can set option 4", False)
        return 1
    m.generate("Say hi.", 4)
    rule = m._set_hold_last
    m._set_hold_last = lambda lv: None                      # the backend sets the option at every request: stop it, so this test can switch it
    text = BASE * 2 + ASK
    r = {0: [], 1: []}
    for _ in range(4):
        for hold, key in ((0, 0), (3, 1)):
            m.lib.set_hold_last(hold)
            s = m.generate(text, 1)["stats"]
            r[key].append(s["prompt_tokens"] / s["seconds"])
    off, on = statistics.median(r[0][1:]), statistics.median(r[1][1:])
    check("prompts read faster with the last token alone (at least 1.15 times)", on > 1.15 * off, f"{off:.1f} against {on:.1f} tokens/s, {on / off:.2f} times")

    # 4. the backend applies the rule when a size is set
    m._set_hold_last = rule
    seen = []
    m._hold_seen = (None, 0.0)
    m.lib.set_hold_last = lambda n: seen.append(n)          # what the backend would tell the engine
    m.hold_last_speeds = lambda lv: (27.6, 7.0)
    m.level = None
    m.set(full["id"])
    check("the backend asks the rule for the size it sets", seen == [facts.hold_last_min(full, 27.6, 7.0)], str(seen))
    seen.clear()
    m.hold_last_speeds = None
    m.level = None
    m._hold_seen = (None, 0.0)
    m.set(full["id"])
    check("with no speeds it uses the named default", seen == [facts.HOLD_LAST_DEFAULT], str(seen))

    # 5. several answers at once: a long prompt and a short one in the same scheduler, with the last token held back. Nobody stalls.
    lib = m.lib
    if lib.has_slots and lib.n_slots >= 2:
        long_ids, short_ids = lib.tokenize(BASE * 6 + ASK), lib.tokenize("Name three colours of the rainbow.")
        steps = {}
        for hold in (0, 3):
            lib.set_hold_last(hold)
            lib.slot_open(0, long_ids)
            lib.slot_open(1, short_ids)
            first, n = {}, 0
            while len(first) < 2 and n < 300:
                n += 1
                for slot, kind, tok in lib.batch_step(64):
                    if kind == 1 and slot not in first:
                        first[slot] = tok
            lib.slot_close(0)
            lib.slot_close(1)
            steps[hold] = (n, len(first))
        check("two prompts in the scheduler both get their first token with the last token held back",
              steps[3][1] == 2 and steps[3][0] <= steps[0][0] + 3, f"steps {steps}")
    else:
        print("only one slot in this library: the scheduler check is skipped")
    return 1 if FAILS else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3] or None, sys.argv[4])
        sys.exit(0)
    sys.exit(main())
