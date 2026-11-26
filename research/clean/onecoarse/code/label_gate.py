"""Label gate for the onecoarse project (Shonu, 2026-11-23: "you AGAIN did it by nouns").

Class labels must never group or select neurons, photos or firings. The 2026-11-22 noun cut (animal / artifact,
ImageNet class index < 398) came in through a pre-run definitions table, a pre-run bar and three scripts, before any
result, so the result lints missed it. This gate checks the places it came in.

Checks
1. Scripts in onecoarse/code/*.py: a code token that loads or uses class labels (the attributes .lab, .label,
   .labels, .targets; the names wnid, synset, class_idx; the string keys "lab", "label", "labels"; the class-index
   cut 398). A local variable named lab (a row title) and the judges' "targets" (target photos) are not labels.
   Comments and docstrings are not checked. A line that ends with "# labels: record-only" passes.
2. onecoarse/LOG.md, only the "## " sections after the line "<!-- label-gate from here -->" (older sections are a
   record):
   a. outside result subsections (pre-run definitions, bars, plans): no class words (animal, artifact, class index,
      the object nouns of verb_lint.py);
   b. in a "###" subsection whose title holds "result": a neuron table (neuron, verb or job, both poles, feeders,
      readers), and no object nouns in its Verb column.
   Lines between "<!-- labels: record-only -->" and "<!-- /labels -->", or a line holding the first marker, pass.

Usage: python3 label_gate.py [paths]     -> lint the files (default: every script and the LOG), exit 1 on problems
       python3 label_gate.py --hook      -> PostToolUse hook on Write / Edit: lints the touched file, exit 2 on problems
       python3 label_gate.py --selftest  -> checks the gate on known bad and clean inputs
"""
import io
import json
import os
import re
import sys
import tempfile
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.normpath(os.path.join(HERE, "..", "LOG.md"))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "neuron", "tools")))
sys.path.insert(0, HERE)
from interp_lint import header_ok, tables  # noqa: E402
from verb_lint import NOUNS  # noqa: E402

MARK = "<!-- labels: record-only -->"
END = "<!-- /labels -->"
CODE_MARK = "# labels: record-only"
ATTRS = {"lab", "label", "labels", "targets"}          # flagged after a dot: photos.lab, batch.targets
NAMES = {"wnid", "wnids", "synset", "synsets", "class_idx"}   # flagged anywhere
KEYS = {"lab", "label", "labels"}                     # flagged as string keys: D["lab"]
CLASS_WORDS = ["animal", "animals", "artifact", "artifacts", "artefact", "artefacts", "class index", "class below",
               "class above", "imagenet class", "living", "man-made"]
OURS = {"child", "head", "heads", "key", "keys"}   # our own words: a child neuron, the classifier head, a dict key
CLASS_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in CLASS_WORDS + [n for n in NOUNS if n not in OURS])
                      + r")\b", re.I)
VERB_BAN = re.compile(r"\b(" + "|".join(NOUNS) + r")\b", re.I)
FROM = "<!-- label-gate from here -->"


def lint_script(text, name="script"):
    probs = []
    lines = text.splitlines()
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as e:
        return [f"{name}: could not tokenize ({e})"]
    prev = None
    for t in toks:
        row = t.start[0]
        before, prev = prev, t
        if row - 1 < len(lines) and CODE_MARK in lines[row - 1]:
            continue
        hit = None
        if t.type == tokenize.NAME and (t.string in NAMES or (t.string in ATTRS and before is not None
                                                                and before.string == ".")):
            hit = t.string
        elif t.type == tokenize.STRING:
            s = t.string.strip("rbuRBU").strip("'\"")
            if s in KEYS:
                hit = t.string
        elif t.type == tokenize.NUMBER and t.string == "398":
            hit = "398"
        if hit:
            probs.append(f"{name}:{row}: class-label use ({hit}): {lines[row - 1].strip()[:90]}")
    return probs


def sections(text):
    parts = re.split(r"(?m)^(## .*)$", text)
    for k in range(1, len(parts), 2):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", parts[k])
        yield parts[k], (m.group(1) if m else None), parts[k + 1]


def unmarked(lines):
    """A line that is only the marker opens a block that runs to the end marker. Any other line that holds the
    marker passes on its own."""
    skip = False
    for i, ln in enumerate(lines):
        if skip:
            skip = END not in ln
            continue
        if MARK in ln:
            skip = ln.strip() == MARK
            continue
        yield i, ln


def verb_col_problems(lines, where):
    probs, col = [], None
    for i, ln in enumerate(lines):
        if not ln.lstrip().startswith("|"):
            col = None
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            col = next((k for k, c in enumerate(cells) if "verb" in c.lower()), None)
            continue
        if col is not None and col < len(cells) and not re.match(r"^[\s:|-]+$", ln) and MARK not in ln:
            hits = sorted(set(m.group(0).lower() for m in VERB_BAN.finditer(cells[col])))
            if hits:
                probs.append(f"{where}: object nouns in the Verb column {hits}: {cells[0][:40]}")
    return probs


def lint_log(text, name="LOG.md"):
    probs = []
    if FROM not in text:
        return probs
    text = text.split(FROM, 1)[1]
    for head, _date, body in sections(text):
        subs = re.split(r"(?m)^(### .*)$", body)
        chunks = [(head, subs[0])] + [(subs[j], subs[j + 1]) for j in range(1, len(subs), 2)]
        for title, chunk in chunks:
            where = f"{name} / {head[3:40]}" + ("" if title is head else f" / {title[4:50]}")
            lines = chunk.splitlines()
            if title is not head and "result" in title.lower():
                if not any(header_ok(h) for h in tables(lines)):
                    probs.append(f"{where}: result with no neuron table (neuron, verb or job, both poles, feeders, readers)")
                probs += verb_col_problems(lines, where)
                continue
            for i, ln in unmarked(lines):
                m = CLASS_RE.search(ln)
                if m:
                    probs.append(f"{where}: class word in pre-run text ({m.group(0)!r}): {ln.strip()[:90]}")
    return probs


def lint_path(path):
    text = open(path).read()
    if path.endswith(".py"):
        return lint_script(text, os.path.basename(path))
    if path.endswith("LOG.md"):
        return lint_log(text, os.path.basename(path))
    return []


def in_scope(path):
    path = os.path.normpath(os.path.abspath(path))
    return (path.startswith(HERE + os.sep) and path.endswith(".py") and os.path.basename(path) != "label_gate.py") \
        or path == LOG


def selftest():
    bad_py = "import numpy as np\nS1 = np.load('x.npz')\nanimal = S1['lab'][0] < 398\n"
    marked_py = "lab = S1['lab']  # labels: record-only\n"
    attr_py = "lab = Photos('read').lab[ix]\n"
    rowtitle_py = "for r, (lab, idx) in enumerate(rows):\n    print(lab)\njson.dump({'targets': tg}, f)\n"
    clean_py = '"""Groups by the input split; no labels or targets used."""\nimport numpy as np\nx = np.zeros(3)  # labels are never loaded\n'
    old = "## Old (2026-11-22)\n\n| Item | Definition |\n|---|---|\n| Animal / artifact | class below 398 = animal |\n"
    old = old + FROM + "\n"
    new_bad = (FROM + "\n## Children again (2026-11-23)\n\n| Item | Definition |\n|---|---|\n"
               "| Animal / artifact | class below 398 = animal |\n\nP1: at least .7 animal photos.\n")
    new_marked = (FROM + "\n## Withdrawn (2026-11-23)\n\n" + MARK + "\n| Status | Item |\n|---|---|\n"
                  "| Withdrawn | the animal / artifact split |\n" + END + "\n")
    res_bad = FROM + "\n## Run (2026-11-23)\n\n### Result\n\nGroups found: 4.\n"
    res_ok = (FROM + "\n## Run (2026-11-23)\n\n### Result\n\n| Neuron | Verb (job) | Positive pole | Negative pole | Feeders | Readers |\n"
              "|---|---|---|---|---|---|\n| n101 | keeps a centre with rays around it | dogs | smooth blobs | s3:69 | n17 |\n")
    res_noun = res_ok.replace("keeps a centre with rays around it", "marks dog faces")
    cases = [
        ("script with lab < 398", lint_script(bad_py), True),
        ("script line marked record-only", lint_script(marked_py), False),
        ("script reading the .lab attribute", lint_script(attr_py), True),
        ("row-title variable and judge targets", lint_script(rowtitle_py), False),
        ("clean script (words in docstring and comment)", lint_script(clean_py), False),
        ("LOG text before the start marker", lint_log(old), False),
        ("LOG with no start marker", lint_log(new_bad.replace(FROM, "")), False),
        ("new pre-run noun cut", lint_log(new_bad), True),
        ("new marked withdrawal note", lint_log(new_marked), False),
        ("new result with no neuron table", lint_log(res_bad), True),
        ("new result with a neuron table", lint_log(res_ok), False),
        ("new result with nouns in the Verb column", lint_log(res_noun), True),
    ]
    real = [os.path.join(HERE, f) for f in ("oc_route.py", "oc_route_check.py", "oc_facets_read.py")]
    ok = True
    for case, probs, want in cases:
        good = bool(probs) == want
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {case}: {'flagged' if probs else 'clean'}")
    for p in real:
        n = len(lint_script(open(p).read(), os.path.basename(p)))
        print(f"info  {os.path.basename(p)}: {n} class-label uses")
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "LOG.md")
        open(f, "w").write(new_bad)
        good = bool(lint_path(f))
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  lint_path on a LOG file")
    print("selftest:", "ok" if ok else "FAILED")
    return 0 if ok else 1


def main():
    if "--selftest" in sys.argv:
        return selftest()
    if "--hook" in sys.argv:
        try:
            call = json.load(sys.stdin)
        except Exception:
            return 0
        path = (call.get("tool_input") or {}).get("file_path", "")
        if not path or not in_scope(path) or not os.path.exists(path):
            return 0
        probs = lint_path(path)
        if probs:
            print("label_gate (Shonu 2026-11-23: labels never group or select; read named neurons):\n"
                  + "\n".join(probs), file=sys.stderr)
            return 2
        return 0
    paths = sys.argv[1:] or [LOG] + sorted(os.path.join(HERE, f) for f in os.listdir(HERE)
                                           if f.endswith(".py") and f != "label_gate.py")
    probs = []
    for p in paths:
        probs += lint_path(p)
    print("\n".join(probs) if probs else "label_gate: ok")
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
