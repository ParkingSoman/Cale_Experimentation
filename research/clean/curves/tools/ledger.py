#!/usr/bin/env python3
"""Check the decision ledger DECISIONS.md against the paper COARSE-THEORY.md.

Usage:
  ledger.py                 check (exit 0 clean, 1 on problems)
  ledger.py --report        print every result with its path and distance from the trunk
  ledger.py --open          print the trunk and the open decisions (for the turn reminder)
  ledger.py --hash          print the hash of the trunk text (to store in trunk.sha256 after Shonu agrees)
Rules:
  K1  every phrase inside an "(under: ...)" tag matches an alias of section 5, or is a joining word
  K2  every tagged item of Parts B and D has a row in the paths table (section 6), and every row names a real item
  K3  every node that an item's tag names is on that item's path
  K4  every node on a path or in an alias exists in the nodes table (section 4)
  K5  every hypothesis declared in Part A.3 has an alias
  K6  statuses: DECIDED rows name Shonu and a date; RETRO rows were added on or before the cut-off date
  K7  the trunk text has the hash stored in tools/trunk.sha256
"""
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CURVES = os.path.dirname(HERE)
LEDGER = os.path.join(CURVES, "DECISIONS.md")
PAPER = os.path.join(CURVES, "COARSE-THEORY.md")
TRUNK_SHA = os.path.join(HERE, "trunk.sha256")
RETRO_CUTOFF = "2026-10-25"
KINDS = {"ROOT", "CONV", "STRUCT", "SCOPE", "NEG"}
STATUSES = {"DECIDED", "PENDING", "RETRO", "DROPPED"}
COUNTED = {"CONV", "STRUCT", "SCOPE"}

ITEM = re.compile(r"^\*\*(Theorem|Lemma|Proposition|Corollary|Conjecture|Remark)\s+([A-Za-z0-9.']+)")
HYP_DECL = re.compile(r"^\*\*Hypothesis (H-[A-Za-z0-9-]+)")
# words that only join phrases inside a tag
JOINERS = re.compile(r"\b(CONJECTURE|none|and|with|the|as (Theorem|Proposition|Lemma|Corollary) [A-Za-z0-9.']+|"
                     r"parts? \d+( (to|and) \d+)?)\b")


def block(text, name):
    m = re.search(rf"<!-- {name} -->\n(.*?)\n<!-- /{name} -->", text, re.S)
    return m.group(1) if m else None


def cells(row):
    parts = re.split(r"(?<!\\)\|", row.strip())[1:-1]
    return [p.strip().replace("\\|", "|") for p in parts]


def table(text, name):
    b = block(text, name)
    if b is None:
        return None
    rows = [cells(r) for r in b.splitlines() if r.strip().startswith("|")]
    return [r for r in rows[2:]]  # drop header and rule


def split_nodes(s):
    s = s.strip()
    if s in ("(none)", ""):
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def load(ledger_text):
    nodes = {}
    for r in table(ledger_text, "nodes") or []:
        if len(r) >= 8:
            nodes[r[0]] = {"kind": r[1], "axis": r[2], "what": r[3], "by": r[5], "added": r[6], "status": r[7]}
    aliases = [(r[0], split_nodes(r[1])) for r in (table(ledger_text, "aliases") or []) if len(r) >= 2]
    aliases.sort(key=lambda a: -len(a[0]))
    paths = {r[0]: split_nodes(r[1]) for r in (table(ledger_text, "paths") or []) if len(r) >= 2}
    return nodes, aliases, paths


def balanced_tag(s):
    i = s.find("(under:")
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(s):
        if s[j] == "(":
            depth += 1
        elif s[j] == ")":
            depth -= 1
            if depth == 0:
                return s[i + len("(under:"):j]
        j += 1
    return s[i + len("(under:"):]


def paper_items(paper_text):
    lines = paper_text.splitlines()
    part, items, hyps = "pre", [], []
    for i, line in enumerate(lines):
        m = re.match(r"^#\s+(Part\s+([A-Z])|Appendix)", line)
        if m:
            part = "Appendix" if m.group(1) == "Appendix" else m.group(2)
        h = HYP_DECL.match(line)
        if h and part == "A":
            hyps.append(h.group(1))
        m = ITEM.match(line)
        if m and part in ("B", "D"):
            head = " ".join(lines[i:i + 6])
            # the tag sits in the bold title, which ends at the first '**' after the opening one
            close = head.find("**", 2)
            title = head[:close] if close > 0 else head
            tag = balanced_tag(title)
            items.append({"name": f"{m.group(1)} {m.group(2)}", "line": i + 1, "tag": tag})
    return items, hyps


def tag_nodes(tag, aliases):
    rest, found = tag, []
    for phrase, ns in aliases:
        if phrase in rest:
            found += ns
            rest = rest.replace(phrase, " ")
    rest = JOINERS.sub(" ", rest)
    rest = re.sub(r"[\s,;:.()]+", " ", rest).strip()
    return [n for n in found if n != "TECH"], rest


def trunk_hash(ledger_text):
    t = block(ledger_text, "trunk") or ""
    return hashlib.sha256(" ".join(t.split()).encode()).hexdigest()


def check(ledger_text, paper_text, sha_path=TRUNK_SHA):
    probs = []
    if block(ledger_text, "trunk") is None:
        probs.append("K7: the trunk markers are missing from DECISIONS.md")
    elif not os.path.exists(sha_path):
        probs.append("K7: tools/trunk.sha256 is missing")
    elif open(sha_path).read().strip() != trunk_hash(ledger_text):
        probs.append("K7: the trunk text changed. Only Shonu changes the trunk. Restore it, or (after Shonu says so) "
                     "store the new hash with: tools/ledger.py --hash > tools/trunk.sha256")
    nodes, aliases, paths = load(ledger_text)
    for nid, n in nodes.items():
        if n["kind"] not in KINDS:
            probs.append(f"K6: node {nid} has unknown kind '{n['kind']}'")
        if n["status"] not in STATUSES:
            probs.append(f"K6: node {nid} has unknown status '{n['status']}'")
        if n["status"] == "DECIDED" and not ("Shonu" in n["by"] and re.search(r"20\d\d-\d\d-\d\d", n["by"])):
            probs.append(f"K6: node {nid} is DECIDED but its 'By' cell does not name Shonu with a date")
        if n["status"] == "RETRO" and n["added"] > RETRO_CUTOFF:
            probs.append(f"K6: node {nid} is RETRO but was added after {RETRO_CUTOFF}. A new node starts PENDING")
    for phrase, ns in aliases:
        for x in ns:
            if x != "TECH" and x not in nodes:
                probs.append(f"K4: alias '{phrase}' points to unknown node {x}")
    items, hyps = paper_items(paper_text)
    names = {it["name"] for it in items}
    for it in items:
        if it["tag"] is None:
            continue
        tn, rest = tag_nodes(it["tag"], aliases)
        if rest:
            probs.append(f"K1 line {it['line']}: {it['name']}: tag phrase '{rest}' has no entry in DECISIONS.md "
                         "section 5. A premise is a node (a decision) or TECH")
        if it["name"] not in paths:
            probs.append(f"K2 line {it['line']}: {it['name']} has no row in the paths table of DECISIONS.md")
            continue
        for x in tn:
            if x not in paths[it["name"]]:
                probs.append(f"K3 line {it['line']}: {it['name']}: its tag names {x}, which is not on its path")
    for name, p in paths.items():
        if name not in names:
            probs.append(f"K2: paths row '{name}' names no tagged item of Parts B or D")
        for x in p:
            if x not in nodes:
                probs.append(f"K4: path of {name} has unknown node {x}")
    alias_text = " ".join(a for a, _ in aliases)
    for h in hyps:
        if not re.search(rf"(?<![\w-]){re.escape(h)}(?![\w-])", alias_text):
            probs.append(f"K5: hypothesis {h} has no alias in DECISIONS.md section 5 (every hypothesis is a node)")
    return probs


def report(ledger_text):
    nodes, _, paths = load(ledger_text)
    rows = []
    for name, p in paths.items():
        d = sum(1 for x in p if nodes.get(x, {}).get("kind") in COUNTED)
        undecided = [x for x in p if nodes.get(x, {}).get("status") in ("PENDING", "RETRO")]
        rows.append((d, name, p, undecided))
    rows.sort()
    out = ["| Item | Distance | Undecided nodes on the path | Path |", "|---|---|---|---|"]
    for d, name, p, und in rows:
        out.append(f"| {name} | {d} | {', '.join(und) or '-'} | {', '.join(p) or '(none)'} |")
    return "\n".join(out)


def open_list(ledger_text):
    nodes, _, _ = load(ledger_text)
    t = " ".join((block(ledger_text, "trunk") or "").split())
    t = re.sub(r"\*\*Trunk T\\\*\.\*\*\s*", "", t)
    pend = [k for k, n in nodes.items() if n["status"] == "PENDING"]
    retro = [k for k, n in nodes.items() if n["status"] == "RETRO"]
    return (f"TRUNK (confirmed by Shonu; only Shonu changes it): {t}\n"
            f"PENDING decisions (Shonu's): {', '.join(pend) or 'none'}. "
            f"RETRO nodes awaiting review: {len(retro)}. Ledger: research/clean/curves/DECISIONS.md\n"
            "Rule 7: no new restriction without a node. A technical node I decide by delegation; a directional one is PENDING until Shonu decides.")


def main():
    lt = open(LEDGER).read() if os.path.exists(LEDGER) else ""
    if "--hash" in sys.argv:
        print(trunk_hash(lt))
        return 0
    if "--report" in sys.argv:
        print(report(lt))
        return 0
    if "--open" in sys.argv:
        print(open_list(lt))
        return 0
    probs = check(lt, open(PAPER).read())
    print("\n".join(probs) if probs else "ledger: clean")
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
