"""Workflows are short files. Code reads the trigger and picks one. A reader follows the steps, and may leave them with a written reason.

File format: a head of "key: value" lines between two lines of ---, then the steps in plain words.
   id: dial
   version: 1
   priority: 20                      lower runs first
   trigger: cliff < .3 and largest > 1
   step_limit: 7
   exits: claim, switch:several_routes, open
   tests: ordering                   adds to what the constitution requires
   replaces_poles: yes               only a dial workflow may use this: an ordering in place of the pole sheets

A trigger may use numbers, the words and / or / not, comparisons, and the names of a sheet's features. It can run no other code."""
import ast
import glob
import operator
import os

OPS = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne}
NEEDED = ("id", "version", "priority", "trigger", "step_limit", "exits")


class BadWorkflow(Exception):
    pass


def evaluate(expr, features):
    def walk(n):
        if isinstance(n, ast.Expression):
            return walk(n.body)
        if isinstance(n, ast.BoolOp):
            vals = [walk(v) for v in n.values]
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return not walk(n.operand)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -walk(n.operand)
        if isinstance(n, ast.Compare):
            left = walk(n.left)
            for op, right in zip(n.ops, n.comparators):
                r = walk(right)
                if type(op) not in OPS or not OPS[type(op)](left, r):
                    return False
                left = r
            return True
        if isinstance(n, ast.Name):
            if n.id in ("True", "true"):
                return True
            if n.id not in features:
                raise BadWorkflow(f"trigger names {n.id}, which is not a sheet feature")
            return features[n.id]
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, bool)):
            return n.value
        raise BadWorkflow(f"a trigger may not hold {type(n).__name__}")
    return bool(walk(ast.parse(expr, mode="eval")))


class Workflow:
    def __init__(self, head, body, path=""):
        missing = [k for k in NEEDED if k not in head]
        if missing:
            raise BadWorkflow(f"{path}: missing {missing}")
        self.id = head["id"]
        self.version = int(head["version"])
        self.priority = int(head["priority"])
        self.trigger = head["trigger"]
        self.step_limit = int(head["step_limit"])
        self.exits = [e.strip() for e in head["exits"].split(",")]
        self.tests = [t.strip() for t in head.get("tests", "").split(",") if t.strip()]
        self.replaces_poles = head.get("replaces_poles", "no").lower() == "yes"
        self.body = body
        self.path = path
        if "open" not in self.exits:
            raise BadWorkflow(f"{path}: every workflow must be able to end open")
        if self.step_limit < 1:
            raise BadWorkflow(f"{path}: a workflow needs a step limit")
        if self.replaces_poles and "ordering" not in self.tests:
            raise BadWorkflow(f"{path}: pole sheets can be replaced only by an ordering test")

    def matches(self, features):
        return evaluate(self.trigger, features)


def parse(text, path=""):
    lines = text.split("\n")
    if lines[0].strip() != "---":
        raise BadWorkflow(f"{path}: no head")
    end = lines[1:].index("---") + 1
    head = {}
    for line in lines[1:end]:
        key, _, value = line.partition(":")
        head[key.strip()] = value.strip()
    return Workflow(head, "\n".join(lines[end + 1:]).strip(), path)


def load(folder):
    out = [parse(open(f, encoding="utf-8").read(), f) for f in sorted(glob.glob(os.path.join(folder, "*.md")))]
    return sorted(out, key=lambda w: (w.priority, w.id))


def route(features, workflows, tried=()):
    """The first workflow by priority whose trigger holds and that this node has not tried. None means nothing fits: the stuck process starts."""
    for w in workflows:
        if w.id not in tried and w.matches(features):
            return w
    return None
