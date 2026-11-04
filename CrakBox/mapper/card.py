"""What a reader must hand in. The tool rejects a card that breaks the rules, so that no brief has to ask twice."""
import re

REQUIRED = ("node", "distinction", "positive", "negative", "shape", "main_parent", "readers", "statements", "prediction", "near_miss", "unread")
SHAPES = {"dial", "detector", "mixed"}
MARKS = {"observed", "inferred"}
FORBIDDEN = re.compile(r"\b(messy|polysemantic|no structure|many weak parents|not clean)\b", re.I)


class BadCard(Exception):
    pass


def check(card, legacy=False, poles=("pos", "neg")):
    """Raises BadCard. A legacy card comes from before this tool: it keeps its words, and it carries no marks. A node with no negative side owes no words for one."""
    required = REQUIRED if not legacy else ("node", "distinction", "positive", "negative")
    if "neg" not in poles:
        required = tuple(k for k in required if k != "negative")
    missing = [k for k in required if k not in card or card[k] == "" or (card[k] is None and k != "main_parent")]
    if missing:
        raise BadCard(f"{card.get('node', '?')}: missing {missing}")
    text = " ".join(str(card.get(k, "")) for k in ("distinction", "positive", "negative", "unread"))
    hit = FORBIDDEN.search(text)
    if hit:
        raise BadCard(f"{card['node']}: '{hit.group(0)}' names no suspect. Write what was tried and what to try next")
    if legacy:
        return
    if card["shape"] not in SHAPES:
        raise BadCard(f"{card['node']}: shape must be one of {sorted(SHAPES)}")
    statements = card["statements"]
    if not isinstance(statements, list) or not statements:
        raise BadCard(f"{card['node']}: statements must be a list of observed or inferred claims")
    for s in statements:
        if s.get("mark") not in MARKS or not s.get("text"):
            raise BadCard(f"{card['node']}: each statement needs text and a mark, one of {sorted(MARKS)}")
    if card["main_parent"] is None and "no parent" not in card["unread"].lower():
        raise BadCard(f"{card['node']}: no main parent, so 'unread' must say 'no parent' and what was tried")
    for r in card.get("routes", []):
        if not r.get("id") or not r.get("inputs"):
            raise BadCard(f"{card['node']}: each route needs an id and its inputs")


def tested(card):
    """Only observed statements count. An inference counts for nothing until it is tested (CONSTITUTION section 7)."""
    return [s for s in card.get("statements", []) if s["mark"] == "observed"]
