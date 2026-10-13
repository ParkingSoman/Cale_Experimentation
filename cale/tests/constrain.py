"""Tests for cale/constrain.py: the checker that keeps a model's answer valid JSON that fits a schema.
Run: python cale/tests/constrain.py     (no model, no server)

The main test is a random walk: build a document one character at a time, always choosing a character the guide allows.
Every finished document must be real JSON and must fit the schema, judged by a separate small validator written here.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cale import constrain as C  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


def valid(schema, v):
    """A small independent JSON-schema checker (the same subset)."""
    if schema in (None, {}, True):
        return True
    if "const" in schema:
        return v == schema["const"] and type(v) is type(schema["const"])
    if "enum" in schema:
        return any(v == e and type(v) is type(e) for e in schema["enum"])
    for k in ("anyOf", "oneOf"):
        if k in schema:
            return any(valid(s, v) for s in schema[k])
    t = schema.get("type")
    if isinstance(t, list):
        return any(valid({**schema, "type": x}, v) for x in t)
    if t == "object" or (t is None and "properties" in schema):
        if not isinstance(v, dict):
            return False
        props = schema.get("properties", {})
        if any(r not in v for r in schema.get("required", [])):
            return False
        if schema.get("additionalProperties") is False or (props and schema.get("additionalProperties") is not True):
            if any(k not in props for k in v):
                return False
        return all(valid(props.get(k, {}), x) for k, x in v.items())
    if t == "array":
        return isinstance(v, list) and schema.get("minItems", 0) <= len(v) <= (schema.get("maxItems") if schema.get("maxItems") is not None else 10 ** 9) \
            and all(valid(schema.get("items", {}), x) for x in v)
    if t == "string":
        return isinstance(v, str) and schema.get("minLength", 0) <= len(v) <= (schema.get("maxLength") if schema.get("maxLength") is not None else 10 ** 9)
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "boolean":
        return isinstance(v, bool)
    if t == "null":
        return v is None
    return True


SCHEMAS = {
    "extraction": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "email": {"type": "string"},
                                                      "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
                   "required": ["name", "age"]},
    "enum + nested": {"type": "object", "properties": {"color": {"enum": ["red", "green", "blue"]},
                                                         "size": {"anyOf": [{"type": "integer"}, {"type": "string", "maxLength": 4}]},
                                                         "pos": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}, "required": ["x", "y"]}},
                      "required": ["color", "pos"]},
    "tool call": {"anyOf": [{"type": "object", "properties": {"name": {"const": "get_weather"}, "arguments": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"enum": ["c", "f"]}}, "required": ["city"]}}, "required": ["name", "arguments"]},
                            {"type": "object", "properties": {"name": {"const": "add"}, "arguments": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}}, "required": ["name", "arguments"]}]},
    "list of numbers": {"type": "array", "items": {"type": "number"}, "minItems": 1, "maxItems": 4},
    "any": {},
    "boolean or null": {"type": ["boolean", "null"]},
}
ALPHABET = list('_{}[]:,"\\ \n-+.0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZtruefalsnl')


def walk(schema, rng, max_len=1500, object_only=False):
    g = C.Guide(schema, object_only=object_only)
    for _ in range(max_len):
        if g.can_end() and rng.random() < 0.6:
            return g.text
        allowed = [c for c in ALPHABET if g.accepts(c)]
        if not allowed:
            return g.text if g.can_end() else None
        g.feed(rng.choice(allowed))
    return g.text if g.can_end() else None


# --- fixed examples
S = SCHEMAS["extraction"]
for text, ok in (('{"name":"Al","age":3}', True), ('{"age":3,"name":"Al"}', True), ('{ "name" : "Al" ,\n "age" : 3 }', True),
                 ('{"name":"Al","age":3,"tags":["a","b"]}', True), ('{"name":"Al"}', False), ('{"name":"Al","age":3.5}', False),
                 ('{"name":1,"age":3}', False), ('{"name":"Al","age":3,"x":1}', False), ('{"name":"Al","age":3,"tags":["a","b","c","d"]}', False),
                 ('[1]', False), ('{"name":"Al","age":3}x', False), ('{"name":"Al","age":03}', False), ('{"name":"A\\u00e9\\n","age":-3}', True),
                 ('{"name":"a\nb","age":1}', False), ('{"name":"Al","age":3,"name":"B"}', False)):
    check(f"extraction {'accepts' if ok else 'refuses'} {text[:48]!r}", C.check(S, text) == ok)
check("a number at the end of the answer counts as finished", C.check({"type": "integer"}, "42") and C.check({"type": "number"}, "-1.5e3"))
check("a half number is not finished", not C.check({"type": "number"}, "1.") and not C.check({"type": "number"}, "-"))
check("enum", C.check({"enum": ["a", 1, None]}, '"a"') and C.check({"enum": ["a", 1, None]}, "1") and not C.check({"enum": ["a", 1]}, '"b"'))
check("const object", C.check({"const": {"a": 1}}, '{"a":1}'))
check("whitespace is capped at 2 in a gap", C.check({}, '{"a":   1}') is False and C.check({}, '{"a":  1}'))
check("blanks after the end are capped", C.check({}, "[1]  ") and not C.check({}, "[1]   "))
check("object_only: a list is refused, an object is fine", not C.Guide(object_only=True).accepts("[") and C.Guide(object_only=True).accepts("{"))
check("minLength/maxLength", C.check({"type": "string", "minLength": 2, "maxLength": 3}, '"ab"') and not C.check({"type": "string", "minLength": 2}, '"a"')
      and not C.check({"type": "string", "maxLength": 1}, '"ab"'))
for bad in ({"type": "frog"}, {"enum": []}, {"anyOf": []}, "x"):
    try:
        C.Guide(bad)
        check(f"a bad schema is refused: {bad!r}", False)
    except C.SchemaError:
        check(f"a bad schema is refused: {bad!r}", True)

# --- limits and enums
big = {"enum": [f"value-{i}" for i in range(300)]}
check("an enum with 300 values: every value is possible, nothing else", all(C.check(big, json.dumps(f"value-{i}")) for i in (0, 63, 64, 150, 299)) and not C.check(big, '"value-300"'))
TOO_BIG = (("an enum of 600 values", {"enum": list(range(600))}), ("33 alternatives", {"anyOf": [{"type": "integer"}] * 33}),
           ("300 properties", {"type": "object", "properties": {f"p{i}": {} for i in range(300)}}),
           ("a 70 KB schema", {"enum": ["x" * 100] * 500 + [{"a": "y" * 70000}]}),
           ("more than 2000 parts", {"anyOf": [{"type": "object", "properties": {f"p{i}": {"type": "string"} for i in range(100)}} for _ in range(30)]}))
for name, bad in TOO_BIG:
    try:
        C.Guide(bad)
        check(f"too big is refused: {name}", False)
    except C.SchemaError:
        check(f"too big is refused: {name}", True)
g = C.Guide({"type": "object", "properties": {"a": {"const": "zzzz"}}, "required": ["a"]})
dead = 0
for _ in range(C.MAX_WIDE + 2):
    try:
        C.next_token(lambda k: [(1, -1.0)], g, lambda i: "q", lambda i: False, 0.0, 1.0, random.Random(1))
    except C.DeadEnd:
        dead += 1
check("the wide search runs only a few times per answer, then a dead end", g.wide_uses == C.MAX_WIDE and dead == C.MAX_WIDE + 2)

# --- prefixes of valid documents are accepted at every step, and a token is accepted whole or not at all
g = C.Guide(S)
doc = '{"name":"Alice","age":30,"tags":["x"]}'
check("every prefix of a valid document is accepted", all(C.Guide(S).accepts(doc[:i]) for i in range(1, len(doc) + 1)))
check("a token that breaks the schema changes nothing", g.feed('{"name":"A') and not g.feed('",3') and g.text == '{"name":"A')
check("tokens of several characters", C.Guide(S).accepts('{"name":"Al","age"') and not C.Guide(S).accepts('{"nam":'))

# --- random walks: every document is real JSON and fits the schema
rng = random.Random(7)
for name, schema in SCHEMAS.items():
    n_ok = n_bad = n_none = 0
    for _ in range(150):
        text = walk(schema, rng)
        if text is None:
            n_none += 1
            continue
        try:
            v = json.loads(text)
        except ValueError:
            n_bad += 1
            continue
        if valid(schema, v):
            n_ok += 1
        else:
            n_bad += 1
    check(f"random walk, {name}: every finished document is valid JSON that fits", n_bad == 0 and n_ok >= 100, (n_ok, n_bad, n_none))
n_ok = sum(1 for _ in range(100) if (t := walk({}, rng, object_only=True)) and isinstance(json.loads(t), dict))
check("random walk, json_object: always an object", n_ok >= 60, n_ok)

print("\nFAILED: " + ", ".join(FAIL) if FAIL else "\nall passed")
sys.exit(1 if FAIL else 0)
