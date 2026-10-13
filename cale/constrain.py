"""Constrained output for both engines: make the model's answer valid JSON that fits a JSON schema. Pure Python.

How: the engine gives the few most likely next tokens. `Guide.accepts(piece)` says whether a token's text can continue a
valid answer. The engine takes the best token that is accepted (or samples among them). A guide keeps the text so far as a set of
possible parser states, so a schema with choices (`anyOf`, `enum`) works. A state is an immutable tuple: copying is free.

Schema support (the part developers use): `type` (object, array, string, number, integer, boolean, null, or a list of types),
`properties`, `required`, `additionalProperties` (an object with `properties` allows only those keys unless it is `true`),
`items`, `minItems`, `maxItems`, `enum`, `const`, `anyOf`, `oneOf`, `minLength`, `maxLength`. Other keywords are ignored.
No schema (`{}` or `None`): any JSON value, `object_only=True` makes it any JSON object (OpenAI's `json_object`).
Whitespace between tokens: at most 2 characters, so the model cannot stall in blanks.
"""
import json

WS = " \n"
MAX_WS = 2
MAX_STATES = 64
# Limits that keep one app's schema from making every token slow for everyone (the model is shared):
MAX_SCHEMA_BYTES = 64_000       # size of one schema as JSON
MAX_NODES = 2_000               # parts of one schema (properties, items, alternatives...)
MAX_ENUM = 512                  # values in one enum
MAX_ALTERNATIVES = 32           # branches of one anyOf / oneOf / type list
MAX_PROPERTIES = 256            # properties of one object
MAX_WIDE = 6                    # times one answer may fall back to the wide candidate search
ESCAPES = '"\\/bfnrtu'


class SchemaError(ValueError):
    pass


# ---- the schema, as nodes -------------------------------------------------------------------------------------------
def node(schema, depth=0, budget=None):
    """A schema as a tuple. Kinds: any, obj, arr, str, num, int, lit, enum, alt. Raises SchemaError for a schema that is too big."""
    if budget is None:
        budget = [MAX_NODES]
        if len(json.dumps(schema, default=str)) > MAX_SCHEMA_BYTES:
            raise SchemaError(f"the schema is larger than {MAX_SCHEMA_BYTES} bytes")
    budget[0] -= 1
    if budget[0] < 0:
        raise SchemaError(f"the schema has more than {MAX_NODES} parts")
    if depth > 24:
        raise SchemaError("the schema is nested too deeply")
    if schema is True or schema is None or schema == {}:
        return ("any",)
    if not isinstance(schema, dict):
        raise SchemaError("a schema must be an object")
    if "const" in schema:
        return ("lit", json.dumps(schema["const"], ensure_ascii=False, separators=(",", ":")))
    if "enum" in schema:
        if not isinstance(schema["enum"], list) or not schema["enum"]:
            raise SchemaError("enum must be a non-empty list")
        if len(schema["enum"]) > MAX_ENUM:
            raise SchemaError(f"an enum may have at most {MAX_ENUM} values")
        return ("enum", tuple(dict.fromkeys(json.dumps(v, ensure_ascii=False, separators=(",", ":")) for v in schema["enum"])))
    for key in ("anyOf", "oneOf"):
        if key in schema:
            if not isinstance(schema[key], list) or not schema[key]:
                raise SchemaError(f"{key} must be a non-empty list")
            if len(schema[key]) > MAX_ALTERNATIVES:
                raise SchemaError(f"{key} may have at most {MAX_ALTERNATIVES} branches")
            return ("alt", tuple(node(x, depth + 1, budget) for x in schema[key]))
    t = schema.get("type")
    if isinstance(t, list):
        if len(t) > MAX_ALTERNATIVES:
            raise SchemaError(f"a type list may have at most {MAX_ALTERNATIVES} types")
        return ("alt", tuple(node({**schema, "type": x}, depth + 1, budget) for x in t))
    if t == "object" or (t is None and "properties" in schema):
        props = schema.get("properties") or {}
        if not isinstance(props, dict) or len(props) > MAX_PROPERTIES:
            raise SchemaError(f"an object may have at most {MAX_PROPERTIES} properties")
        required = tuple(k for k in schema.get("required", []) if k in props)
        if any(k not in props for k in schema.get("required", [])) and schema.get("additionalProperties") is False:
            raise SchemaError("a required property is not in properties")
        extra = schema.get("additionalProperties") is True or (not props and schema.get("additionalProperties") is not False)
        return ("obj", tuple((k, node(v, depth + 1, budget)) for k, v in props.items()), required, extra)
    if t == "array":
        return ("arr", node(schema.get("items", {}), depth + 1, budget), int(schema.get("minItems", 0)), schema.get("maxItems"))
    if t == "string":
        return ("str", int(schema.get("minLength", 0)), schema.get("maxLength"))
    if t == "integer":
        return ("int",)
    if t == "number":
        return ("num",)
    if t == "boolean":
        return ("enum", ("true", "false"))
    if t == "null":
        return ("lit", "null")
    if t is None:
        return ("any",)
    raise SchemaError(f"unknown type {t!r}")


# ---- the parser: a state is a stack (tuple) of frames; a frame is a tuple. `step` returns the stacks after one character ----
# frames:
#   ("value", node, ws)                      a value of this schema is expected (ws = blanks seen so far in this gap)
#   ("lit", text, pos)                       inside a literal, `pos` characters matched
#   ("enumv", texts, pos)                    inside one of several literals: `texts` are those that still match the first `pos` characters
#   ("str", minlen, maxlen, n, esc)          inside a string: n characters so far, esc = "" | "\\" | "u0".."u3" (hex digits left)
#   ("key", props, seen, text, extra)        inside an object key
#   ("num", integer, text)                   inside a number
#   ("obj_open", node, seen, ws, comma)      after '{' (comma False: a key or '}') or after ',' (comma True: a key only)
#   ("obj_colon", node, seen, key, ws)       after a key: ':' expected
#   ("obj_next", node, seen, ws)             after a value: ',' or '}'
#   ("arr_open", node, count, ws)            after '[': a value or ']'
#   ("arr_next", node, count, ws)            after a value: ',' or ']'
def start(schema=None, object_only=False, budget=MAX_NODES):
    n = ("anyobj",) if (object_only and (schema is None or schema == {})) else node(schema, 0, [budget] if budget != MAX_NODES else None)
    return [(("value", n, 0),)]


def _num_ok(text, integer):
    """Is `text` a complete number? A valid prefix is checked by _num_step."""
    try:
        float(text)
    except ValueError:
        return False
    return text[-1].isdigit() and not (text[0] == "0" and len(text) > 1 and text[1].isdigit()) and not (text.startswith("-0") and len(text) > 2 and text[2].isdigit())


def _num_step(text, ch, integer):
    nxt = text + ch
    if ch.isdigit():
        if text in ("0", "-0"):
            return None                                           # no leading zeros
        return nxt
    if ch == "-" and text == "":
        return nxt
    if integer:
        return None
    if ch == "." and text and text[-1].isdigit() and "." not in text and "e" not in text.lower():
        return nxt
    if ch in "eE" and text and text[-1].isdigit() and "e" not in text.lower():
        return nxt
    if ch in "+-" and text and text[-1] in "eE":
        return nxt
    return None


def step(stack, ch):
    """All stacks that can follow `stack` after the character `ch` (a list, empty if `ch` is not allowed)."""
    if not stack:
        return []
    top = stack[-1]
    kind = top[0]
    if kind == "done":                                              # the answer is complete: only a few blanks may follow
        return [(("done", top[1] + 1),)] if ch in WS and top[1] < MAX_WS else []
    if kind == "value":
        return _step_value(stack, top, ch)
    if kind == "lit":
        _, text, pos = top
        if ch != text[pos]:
            return []
        if pos + 1 < len(text):
            return [stack[:-1] + (("lit", text, pos + 1),)]
        return [_complete(stack[:-1])]
    if kind == "enumv":
        _, texts, pos = top
        left = tuple(t for t in texts if len(t) > pos and t[pos] == ch)
        out = []
        if any(len(t) == pos + 1 for t in left):                                   # a value ends here
            out.append(_complete(stack[:-1]))
        longer = tuple(t for t in left if len(t) > pos + 1)
        if longer:                                                                  # ... and others go on
            out.append(stack[:-1] + (("enumv", longer, pos + 1),))
        return out
    if kind == "str":
        return _step_str(stack, top, ch)
    if kind == "key":
        return _step_key(stack, top, ch)
    if kind == "num":
        _, integer, text = top
        nxt = _num_step(text, ch, integer)
        if nxt is not None:
            return [stack[:-1] + (("num", integer, nxt),)]
        if _num_ok(text, integer):                                 # the number ended: this character belongs to the parent
            return step(_complete(stack[:-1]), ch)
        return []
    if kind == "obj_open":
        _, n, seen, ws, comma = top
        if ch in WS:
            return [stack[:-1] + (("obj_open", n, seen, ws + 1, comma),)] if ws < MAX_WS else []
        required = n[2]
        if ch == "}" and not comma and all(r in seen for r in required):
            return [_complete(stack[:-1])]
        if ch == '"':
            free = tuple(k for k, _ in n[1] if k not in seen)
            if free or n[3]:
                return [stack[:-1] + (("key", n, seen, "", n[3]),)]
        return []
    if kind == "obj_colon":
        _, n, seen, key, ws = top
        if ch in WS:
            return [stack[:-1] + (("obj_colon", n, seen, key, ws + 1),)] if ws < MAX_WS else []
        if ch == ":":
            sub = dict(n[1]).get(key, ("any",))
            return [stack[:-1] + (("obj_colon_val", n, seen | {key}), ("value", sub, 0))]
        return []
    if kind == "obj_next":
        _, n, seen, ws = top
        if ch in WS:
            return [stack[:-1] + (("obj_next", n, seen, ws + 1),)] if ws < MAX_WS else []
        if ch == ",":
            if any(k not in seen for k, _ in n[1]) or n[3]:
                return [stack[:-1] + (("obj_open", n, seen, 0, True),)]
            return []
        if ch == "}" and all(r in seen for r in n[2]):
            return [_complete(stack[:-1])]
        return []
    if kind == "arr_open":
        _, n, count, ws = top
        if ch in WS:
            return [stack[:-1] + (("arr_open", n, count, ws + 1),)] if ws < MAX_WS else []
        if ch == "]" and n[2] <= 0:
            return [_complete(stack[:-1])]
        return _step_value(stack[:-1] + (("arr_val", n, count), ("value", n[1], 0)), ("value", n[1], 0), ch)
    if kind == "arr_next":
        _, n, count, ws = top
        if ch in WS:
            return [stack[:-1] + (("arr_next", n, count, ws + 1),)] if ws < MAX_WS else []
        if ch == "," and (n[3] is None or count < n[3]):
            return [stack[:-1] + (("arr_val", n, count), ("value", n[1], 0))]
        if ch == "]" and count >= n[2]:
            return [_complete(stack[:-1])]
        return []
    return []


def _complete(stack):
    """A value finished: pass it to the parent frame (an object waiting for its next key or comma, an array, or the end)."""
    if not stack:
        return (("done", 0),)
    top = stack[-1]
    if top[0] == "obj_colon_val":
        return stack[:-1] + (("obj_next", top[1], top[2], 0),)
    if top[0] == "arr_val":
        return stack[:-1] + (("arr_next", top[1], top[2] + 1, 0),)
    return stack


def _step_value(stack, top, ch):
    _, n, ws = top
    base = stack[:-1]
    kind = n[0]
    if kind == "alt":
        out = []
        for sub in n[1]:
            out += step(base + (("value", sub, ws),), ch)
        return out
    if ch in WS:
        return [base + (("value", n, ws + 1),)] if ws < MAX_WS else []
    if kind == "any" or kind == "anyobj":
        if kind == "anyobj" and ch != "{":
            return []
        if ch == "{":
            return [base + (("obj_open", ("obj", (), (), True), frozenset(), 0, False),)]
        if ch == "[":
            return [base + (("arr_open", ("arr", ("any",), 0, None), 0, 0),)]
        if ch == '"':
            return [base + (("str", 0, None, 0, ""),)]
        if ch == "-" or ch.isdigit():
            return step(base + (("value", ("num",), 0),), ch)
        for text in ("true", "false", "null"):
            if ch == text[0]:
                return [base + (("lit", text, 1),)]
        return []
    if kind == "obj":
        return [base + (("obj_open", n, frozenset(), 0, False),)] if ch == "{" else []
    if kind == "arr":
        return [base + (("arr_open", n, 0, 0),)] if ch == "[" else []
    if kind == "str":
        return [base + (("str", n[1], n[2], 0, ""),)] if ch == '"' else []
    if kind in ("num", "int"):
        nxt = _num_step("", ch, kind == "int")
        return [base + (("num", kind == "int", nxt),)] if nxt is not None else []
    if kind == "enum":
        return step(base + (("enumv", n[1], 0),), ch)
    if kind == "lit":
        if ch != n[1][0]:
            return []
        return [base + (("lit", n[1], 1),)] if len(n[1]) > 1 else [_complete(base)]
    return []


def _step_str(stack, top, ch):
    _, minlen, maxlen, n, esc = top
    base = stack[:-1]
    if esc.startswith("u"):
        if ch in "0123456789abcdefABCDEF":
            left = int(esc[1:]) - 1
            return [base + (("str", minlen, maxlen, n, f"u{left}" if left else ""),)] if left else [base + (("str", minlen, maxlen, n + 1, ""),)]
        return []
    if esc == "\\":
        if ch == "u":
            return [base + (("str", minlen, maxlen, n, "u4"),)]
        if ch in ESCAPES:
            return [base + (("str", minlen, maxlen, n + 1, ""),)]
        return []
    if ch == "\\":
        return [base + (("str", minlen, maxlen, n, "\\"),)] if maxlen is None or n < maxlen else []
    if ch == '"':
        return [_complete(base)] if n >= minlen else []
    if ord(ch) < 0x20:
        return []
    if maxlen is not None and n >= maxlen:
        return []
    return [base + (("str", minlen, maxlen, n + 1, ""),)]


def _step_key(stack, top, ch):
    _, n, seen, text, extra = top
    base = stack[:-1]
    if ch == '"':
        if text in {k for k, _ in n[1] if k not in seen} or (extra and text and text not in dict(n[1])):
            return [base + (("obj_colon", n, seen, text, 0),)]
        return []
    if ch == "\\" or ord(ch) < 0x20:
        return []
    cand = text + ch
    if extra or any(k.startswith(cand) for k, _ in n[1] if k not in seen):
        return [base + (("key", n, seen, cand, extra),)]
    return []


# ---- the guide ------------------------------------------------------------------------------------------------------
class Guide:
    """The text so far, as the set of states it may be in. `accepts(piece)` tests a token. `feed(piece)` takes it."""

    def __init__(self, schema=None, object_only=False, states=None, text="", budget=MAX_NODES):
        self.states = states if states is not None else start(schema, object_only, budget)
        self.text = text
        self.wide_uses = 0                                                  # how often the expensive wide search was needed

    def _run(self, piece):
        states = self.states
        for ch in piece:
            nxt, seen = [], set()
            for s in states:
                for t in step(s, ch):
                    if t not in seen:
                        seen.add(t)
                        nxt.append(t)
                if len(nxt) > MAX_STATES:
                    break
            if not nxt:
                return None
            states = nxt[:MAX_STATES]
        return states

    def accepts(self, piece):
        return bool(piece) and self._run(piece) is not None

    def feed(self, piece):
        """Take the text. Returns False (and changes nothing) if it cannot continue a valid answer."""
        s = self._run(piece)
        if s is None:
            return False
        self.states, self.text = s, self.text + piece
        return True

    def can_end(self):
        """True if the text so far is a complete value. Then the only thing left to do is stop."""
        return any(len(s) == 1 and (s[0][0] == "done" or (s[0][0] == "num" and _num_ok(s[0][2], s[0][1]))) for s in self.states)

    def finished(self):
        """True if nothing but blanks or the end can follow."""
        return all(len(s) == 1 and s[0][0] == "done" for s in self.states)


def check(schema, text):
    """Does `text` (a whole JSON text) fit `schema`? Used by tests and to validate tool-call arguments."""
    g = Guide(schema)
    return g.feed(text) and g.can_end()


# ---- choosing a token ---------------------------------------------------------------------------------------------
K_FIRST = 64        # candidates looked at first. Almost always one of them fits.
K_WIDE = 4096       # if none fits, look at many more before giving up


def next_token(candidates, guide, text_of, is_eog, temperature, top_p, rng):
    """The next token for a guided answer. `candidates(k)` gives the k most likely [(id, log-probability)].
    First K_FIRST are tried; if none fits, K_WIDE, but at most MAX_WIDE times per answer (it is slow: a strict schema must not stall everyone).
    Returns (id, text or None). Raises DeadEnd."""
    pick = choose_token(candidates(K_FIRST), guide, text_of, is_eog, temperature, top_p, rng)
    if pick is None and guide.wide_uses < MAX_WIDE:
        guide.wide_uses += 1
        pick = choose_token(candidates(K_WIDE), guide, text_of, is_eog, temperature, top_p, rng)
    if pick is None:
        raise DeadEnd("no token of this model can continue a valid answer")
    return pick


class DeadEnd(RuntimeError):
    """No token of the model can continue a valid answer (a very strict schema, or a token set that cannot spell it)."""


def choose_token(cands, guide, text_of, is_eog, temperature, top_p, rng):
    """Pick the next token. cands: [(token id, log-probability)], best first. text_of(id) -> str or None (None: never allowed).
    is_eog(id): the end-of-answer tokens, allowed only when the answer is complete.
    temperature 0: the best candidate that fits. Otherwise a sample among the candidates that fit, by probability.
    Returns (id, text) with text None for an end token, or None if nothing fits."""
    ok = []
    for tid, lp in cands:
        if is_eog(tid):
            if guide.can_end():
                ok.append((tid, lp, None))
        else:
            t = text_of(tid)
            if t and guide.accepts(t):
                ok.append((tid, lp, t))
        if ok and temperature <= 0:
            break
    if not ok:
        return None
    if temperature <= 0 or len(ok) == 1:
        return ok[0][0], ok[0][2]
    import math
    w = [math.exp((lp - ok[0][1]) / temperature) for _, lp, _ in ok]
    tot = sum(w)
    p = [x / tot for x in w]
    if top_p < 1.0:                                              # the smallest set whose probability reaches top_p (always the best)
        keep, acc = 0, 0.0
        for i, x in enumerate(p):
            keep, acc = i + 1, acc + x
            if acc >= top_p:
                break
        p = [x / sum(p[:keep]) for x in p[:keep]]
        ok = ok[:keep]
    r, acc = rng.random(), 0.0
    for (tid, _, t), x in zip(ok, p):
        acc += x
        if r <= acc:
            return tid, t
    return ok[-1][0], ok[-1][2]
