"""Stop hook (Shonu's rule, 2026-11-21: "readings are never objectively messy"). Reads the hook input on stdin, finds
the last assistant message in the transcript, and blocks the stop once if it calls a neuron messy / polysemantic /
scattered / incoherent / "no clear job" without naming the tool gap. The reminder asks for a rewrite in microscope terms.
Register (Shonu, after the CrakBox unlock): a "Stop" hook running
  python3 "$CLAUDE_PROJECT_DIR/research/clean/neuron/tools/no_messy_stop.py"
"""
import json
import re
import sys

BANNED = re.compile(r"\b(messy|polysemantic|scattered|incoherent|mixed bag|no clear (?:job|theme|pattern)|not clean)\b", re.I)


def last_assistant_text(path):
    text = ""
    for line in open(path):
        try:
            d = json.loads(line)
        except Exception:
            continue
        msg = d.get("message") or {}
        if d.get("type") == "assistant" or msg.get("role") == "assistant":
            parts = msg.get("content") or []
            t = " ".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text")
            if t.strip():
                text = t
    return text


def main():
    try:
        inp = json.load(sys.stdin)
    except Exception:
        return 0
    if inp.get("stop_hook_active"):
        return 0
    try:
        text = last_assistant_text(inp["transcript_path"])
    except Exception:
        return 0
    hits = sorted({m.group(0).lower() for m in BANNED.finditer(text)})
    if hits and "tool gap" not in text.lower():
        print(json.dumps({"decision": "block", "reason":
            f"Standing rule (2026-11-21): you called a reading {hits}. A neuron is never messy; the microscope failed. "
            "Rewrite that part as 'my reading failed; tool gap: <which>' and name the tool fix (peak-centred crops, "
            "both poles, the photo's path one layer back, more photos, checkpoints)."}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
