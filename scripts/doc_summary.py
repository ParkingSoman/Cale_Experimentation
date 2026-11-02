"""Read the output of `chat_race.py --docs` on standard input and print one line per file type: OK or FAIL. Exit 1 if a type (other than the known-bad EPUB sample) fails."""
import json
import sys

text = sys.stdin.read()
if "document check:" not in text:
    print("no document check in the output")
    sys.exit(1)
d = json.loads(text.split("document check:")[1].splitlines()[0])
bad = [k for k, v in d.items() if isinstance(v, dict) and not v.get("knows_the_fact")]
print(" ".join(k + ":" + ("OK" if k not in bad else "FAIL") for k, v in d.items() if isinstance(v, dict)))
for k in bad:
    print("  ", k, str(d[k])[:160])
sys.exit(1 if [b for b in bad if b != "log.epub"] else 0)
