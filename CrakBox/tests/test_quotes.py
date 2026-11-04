"""Checks the two documents. Every quote in PRINCIPLES.md must sit on the essay line it names. CONSTITUTION.md must name no network.
Usage: test_quotes.py. Exit 0 when clean, 1 with one line per problem."""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ESSAY = os.path.join(ROOT, "..", "research", "clean", "curves", "map_agents", "essay.md")
QUOTE = re.compile(r'^L(\d+): "(.+)"\s*$')
NETWORK_WORDS = re.compile(r"resnet|convnext|inception|gemma|smollm|qwen|stage \d|imagenet", re.I)


def plain(text):
    for curly, straight in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"')):
        text = text.replace(curly, straight)
    return text


def check_quotes(doc_lines, essay_lines):
    problems = []
    count = 0
    for line in doc_lines:
        m = QUOTE.match(line.strip())
        if not m:
            continue
        count += 1
        number, words = int(m.group(1)), plain(m.group(2))
        target = plain(essay_lines[number - 1]) if 0 < number <= len(essay_lines) else ""
        if words not in target:
            problems.append(f"L{number}: not on that line: {words[:80]}")
    return count, problems


def check_constitution(lines):
    return [f"CONSTITUTION line {i}: names a network: {line.strip()[:80]}"
            for i, line in enumerate(lines, 1) if NETWORK_WORDS.search(line)]


def self_test():
    count, problems = check_quotes(['L1: "wrong words"', 'L1: "right words"'], ["these are right words here"])
    assert count == 2 and len(problems) == 1, "the quote check must catch a wrong quote and pass a right one"
    count, problems = check_quotes(['L1: "it\'s fine"'], ["it’s fine"])
    assert count == 1 and not problems, "a curly apostrophe must match a straight one"
    assert check_constitution(["uses ResNet here"]) and not check_constitution(["a network"]), "the network scan must work"


def main():
    self_test()
    essay = open(ESSAY, encoding="utf-8").read().split("\n")
    principles = open(os.path.join(ROOT, "PRINCIPLES.md"), encoding="utf-8").read().split("\n")
    constitution = open(os.path.join(ROOT, "CONSTITUTION.md"), encoding="utf-8").read().split("\n")
    count, problems = check_quotes(principles, essay)
    problems += check_constitution(constitution)
    for p in problems:
        print(p)
    print(f"{count} quotes checked, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
