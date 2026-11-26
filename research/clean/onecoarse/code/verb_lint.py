"""Verb lint for onecoarse/LOG.md (README reading rule 3). In every markdown table after the marker line
'<!-- verb-lint from here -->', the cells of a column whose header contains 'Verb' must hold no object, animal,
person or body-part nouns. Prints each offending cell; exit 1 if any.
Usage: python3 verb_lint.py [path]"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "LOG.md")
MARK = "<!-- verb-lint from here -->"
NOUNS = r"""people person persons man men woman women boy girl baby child human humans player
dog dogs puppy cat cats kitten bird birds fox wolf wolves animal animals monkey primate lion horse snake insect spider
face faces head heads nose noses muzzle snout mouth eye eyes ear ears beak torso neck chest skin fur hair coat paw
wheel wheels hub hubs tyre tire tires car cars vehicle vehicles truck bicycle bike rim rims spoke spokes
flower flowers daisy petal petals dome domes roof clock clocks dial dials bowl bowls bottle bottles jug mug cup
compass fan flag tie collar knob hook bell umbrella plunger chain keyboard key keys""".split()
BAN = re.compile(r"\b(" + "|".join(NOUNS) + r")\b", re.I)


def main(path):
    text = open(path).read()
    if MARK not in text:
        print("verb_lint: no marker line; nothing checked")
        return 0
    lines = text.split(MARK, 1)[1].split("\n")
    bad, col = [], None
    for i, ln in enumerate(lines):
        if not ln.lstrip().startswith("|"):
            col = None
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            col = next((k for k, c in enumerate(cells) if "verb" in c.lower()), None)
            continue
        if col is not None and col < len(cells) and not re.match(r"^[\s:|-]+$", ln):
            hits = sorted(set(m.group(0).lower() for m in BAN.finditer(cells[col])))
            if hits:
                bad.append(f"  {cells[0][:40]!r}: {hits}")
    if bad:
        print("verb_lint: object nouns in the Verb (job) column (README rule 3):")
        print("\n".join(bad))
        return 1
    print("verb_lint: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else LOG))
