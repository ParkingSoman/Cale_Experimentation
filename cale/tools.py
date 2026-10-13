"""Tools and structured output on top of any engine. Pure Python.

Gemma 3n has no built-in tool calling, so Cale does it in the prompt and checks it with the guide (constrain.py):

| Request | What Cale does |
|---|---|
| `response_format` json_object / json_schema | the answer is guided: it is valid JSON, and fits the schema |
| `tools`, `tool_choice` "required" or one named tool | the answer is guided: it is `{"tool_calls": [{"name", "arguments"}]}` and every `arguments` fits that tool's `parameters` |
| `tools`, `tool_choice` "auto" (the default) | the model decides. It is told to answer with that JSON object only when it needs a tool. Cale reads the reply. If it tried to call a tool and got it wrong, Cale asks again with the answer guided. |
| `tool_choice` "none" | tools are not shown to the model |
| assistant `tool_calls` and `tool` messages in the history | written into the conversation as text, in the same JSON shape |

The shapes follow the OpenAI chat API, so existing tool-using code works unchanged.
"""
import json
import re
import uuid

from . import constrain, media

NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_TOOLS = 64


class ToolError(ValueError):
    def __init__(self, code, message, param=None):
        super().__init__(message)
        self.code, self.param = code, param


# ---- reading the request ----------------------------------------------------------------------------------------------
def parse_tools(tools):
    """[{"type": "function", "function": {"name", "description", "parameters"}}] -> [{"name", "description", "parameters"}]."""
    if tools is None:
        return []
    if not isinstance(tools, list) or len(tools) > MAX_TOOLS:
        raise ToolError("invalid_tools", f"'tools' must be a list of at most {MAX_TOOLS} tools.", "tools")
    out, seen = [], set()
    for t in tools:
        if not isinstance(t, dict) or t.get("type") != "function" or not isinstance(t.get("function"), dict):
            raise ToolError("invalid_tools", "Each tool must be {\"type\": \"function\", \"function\": {...}}.", "tools")
        f = t["function"]
        name = f.get("name")
        if not isinstance(name, str) or not NAME.match(name):
            raise ToolError("invalid_tools", "A tool name must be 1 to 64 letters, digits, '_' or '-'.", "tools")
        if name in seen:
            raise ToolError("invalid_tools", f"Two tools are named '{name}'.", "tools")
        seen.add(name)
        params = f.get("parameters") or {"type": "object", "properties": {}}
        try:
            constrain.node(params)
            if len(json.dumps(params, default=str)) > 16_000:
                raise constrain.SchemaError("the parameters are larger than 16000 bytes")
        except constrain.SchemaError as e:
            raise ToolError("invalid_tools", f"The parameters of '{name}' are not a schema Cale can use: {e}", "tools")
        out.append({"name": name, "description": str(f.get("description") or "")[:600], "parameters": params})
    return out


def parse_choice(choice, tools):
    """-> ("none" | "auto" | "required" | the name of one tool)."""
    if choice is None:
        return "auto" if tools else "none"
    if not tools:
        raise ToolError("invalid_tool_choice", "'tool_choice' needs 'tools'.", "tool_choice")
    if choice in ("none", "auto", "required"):
        return choice
    if isinstance(choice, dict) and choice.get("type") == "function" and isinstance(choice.get("function"), dict):
        name = choice["function"].get("name")
        if name in {t["name"] for t in tools}:
            return name
        raise ToolError("invalid_tool_choice", f"'tool_choice' names '{name}', which is not in 'tools'.", "tool_choice")
    raise ToolError("invalid_tool_choice", "'tool_choice' must be \"none\", \"auto\", \"required\" or {\"type\": \"function\", \"function\": {\"name\": ...}}.", "tool_choice")


def response_guide(fmt):
    """The guide and the instruction for a `response_format`, or (None, None) for plain text."""
    if fmt is None or (isinstance(fmt, dict) and fmt.get("type") == "text"):
        return None, None
    if not isinstance(fmt, dict):
        raise ToolError("invalid_response_format", "'response_format' must be an object.", "response_format")
    if fmt.get("type") == "json_object":
        return constrain.Guide(object_only=True), "Reply with a single JSON object and nothing else."
    if fmt.get("type") == "json_schema":
        js = fmt.get("json_schema")
        schema = js.get("schema") if isinstance(js, dict) else None
        if not isinstance(schema, dict):
            raise ToolError("invalid_response_format", "A json_schema format needs {\"json_schema\": {\"name\": ..., \"schema\": {...}}}.", "response_format")
        try:
            guide = constrain.Guide(schema)
        except constrain.SchemaError as e:
            raise ToolError("invalid_schema", f"Cale cannot use this schema: {e}", "response_format")
        return guide, "Reply with a single JSON value that follows this JSON schema, and nothing else:\n" + json.dumps(schema, ensure_ascii=False)
    raise ToolError("invalid_response_format", "'response_format.type' must be \"text\", \"json_object\" or \"json_schema\".", "response_format")


# ---- the conversation -------------------------------------------------------------------------------------------------
def call_text(calls):
    """Tool calls (OpenAI shape) written as the text the model is asked to produce."""
    items = []
    for c in calls:
        f = c.get("function", {})
        args = f.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass
        items.append({"name": f.get("name"), "arguments": args})
    return json.dumps({"tool_calls": items}, ensure_ascii=False, separators=(",", ":"))


def convert_messages(raw, ApiError, images_ok=False):
    """OpenAI messages (with tool_calls and tool roles) -> plain {"role", "content"} messages. `ApiError(status, code, message, param=)` is the server's."""
    if not isinstance(raw, list) or not raw:
        raise ApiError(400, "invalid_messages", "'messages' must be a non-empty list.")
    out, names = [], {}
    for m in raw:
        if not isinstance(m, dict):
            raise ApiError(400, "invalid_messages", "Each message must be an object.")
        role, c = m.get("role"), m.get("content")
        if role == "developer":
            role = "system"
        if role not in ("system", "user", "assistant", "tool"):
            raise ApiError(400, "invalid_messages", f"Role '{role}' is not supported.")
        parts = None
        if isinstance(c, list):
            parts, texts = [], []
            for p in c:
                kind = p.get("type") if isinstance(p, dict) else None
                if kind == "text":
                    texts.append(p.get("text", ""))
                    parts.append(("text", p.get("text", "")))
                elif kind == "image_url":
                    if not images_ok:
                        raise ApiError(400, "unsupported_content", "This computer's engine cannot read pictures yet. "
                                       "'cale.capabilities.images' in GET /v1/models says whether it can.")
                    if role != "user":
                        raise ApiError(400, "unsupported_content", "Pictures are accepted in user messages only.")
                    try:
                        parts.append(("image", media.image_from_part(p)))
                    except media.MediaError as e:
                        raise ApiError(400, "invalid_image", str(e))
                elif kind in ("input_audio", "audio_url", "video_url", "file"):
                    raise ApiError(400, "unsupported_content", f"Content of type '{kind}' is not supported. Supported: text" + (", image_url." if images_ok else "."))
                else:
                    raise ApiError(400, "unsupported_content", "Only text content" + (" and pictures are" if images_ok else " is") + " supported.")
            c = "\n".join(texts)
            if sum(1 for q in parts if q[0] == "image") == 0:
                parts = None
        if c is None and role == "assistant" and m.get("tool_calls"):
            c = ""
        if not isinstance(c, str):
            raise ApiError(400, "invalid_messages", "Message content must be text.")
        if role == "assistant" and m.get("tool_calls"):
            calls = m["tool_calls"]
            if not isinstance(calls, list) or any(not isinstance(x, dict) for x in calls):
                raise ApiError(400, "invalid_messages", "'tool_calls' must be a list of objects.")
            for x in calls:
                names[x.get("id")] = x.get("function", {}).get("name")
            c = (c + "\n" if c else "") + call_text(calls)
        if role == "tool":
            who = names.get(m.get("tool_call_id")) or m.get("name") or "a tool"
            out.append({"role": "user", "content": f"Result of {who} (call {m.get('tool_call_id', '?')}):\n{c}\n\n"
                                                    "(That is the tool's result. Use it to answer my question in plain text. Call another tool only if you still need one.)"})
            continue
        msg = {"role": role, "content": c}
        if parts:
            msg["parts"] = parts
        out.append(msg)
    out = merge_same_role(out)
    if sum(1 for m in out for q in m.get("parts", ()) if q[0] == "image") > media.MAX_IMAGES:
        raise ApiError(400, "invalid_image", f"A request may hold at most {media.MAX_IMAGES} pictures.")
    return out


def merge_same_role(msgs):
    """The model's chat template needs user and assistant turns to alternate. Two turns of the same role in a row (several tool results,
    or two user messages) become one turn, in order. Apps do this all the time with the OpenAI API, which allows it.
    One pass: each group is collected first and joined once, so the cost grows with the size of the request, not its square."""
    groups = []
    for m in msgs:
        if groups and groups[-1][0]["role"] == m["role"] and m["role"] != "system":
            groups[-1].append(m)
        else:
            groups.append([m])
    out = []
    for g in groups:
        if len(g) == 1:
            out.append(dict(g[0]))
            continue
        merged = {"role": g[0]["role"], "content": "\n\n".join(m["content"] for m in g).strip()}
        if any(m.get("parts") for m in g):
            parts = []
            for k, m in enumerate(g):
                if k:
                    parts.append(("text", "\n\n"))
                parts.extend(m.get("parts") or [("text", m["content"])])
            merged["parts"] = parts
        out.append(merged)
    return out


def tool_prompt(tools, choice, parallel=True):
    """The instruction that tells the model what tools exist and how to call them."""
    shown = json.dumps([{"name": t["name"], "description": t["description"], "parameters": t["parameters"]} for t in tools], ensure_ascii=False)
    must = choice != "auto"
    how = ('To call tools, reply with ONLY a JSON object of this shape and nothing else: '
           '{"tool_calls": [{"name": "<tool name>", "arguments": {<arguments>}}]}. ')
    if parallel:
        how += "You may call several tools at once by adding more items. "
    if choice not in ("auto", "required"):
        how += f"You must call the tool '{choice}'. "
    elif must:
        how += "You must call a tool. "
    else:
        how += ("Call a tool only when the request needs something one of the tools provides (live data, a calculation, an action). "
                "For general knowledge, explanations, jokes and conversation, do not call a tool: answer in plain text. "
                "After a tool result arrives, use it to answer. ")
    return "You can use these tools:\n" + shown + "\n" + how


def with_system(messages, text):
    """The messages with `text` added to the first system message (or as a new one at the start)."""
    if not text:
        return messages
    if messages and messages[0]["role"] == "system":
        return [{"role": "system", "content": (messages[0]["content"] + "\n\n" + text).strip()}] + messages[1:]
    return [{"role": "system", "content": text}] + messages


def tool_guide(tools, choice, parallel=True):
    """A guide for an answer that must be a tool call (choice "required" or one tool's name)."""
    chosen = tools if choice == "required" else [t for t in tools if t["name"] == choice]
    items = [{"type": "object", "properties": {"name": {"const": t["name"]}, "arguments": t["parameters"]}, "required": ["name", "arguments"]} for t in chosen]
    schema = {"type": "object", "properties": {"tool_calls": {"type": "array", "minItems": 1, **({} if parallel else {"maxItems": 1}),
                                                              "items": items[0] if len(items) == 1 else {"anyOf": items}}},
              "required": ["tool_calls"]}
    return constrain.Guide(schema, budget=constrain.MAX_NODES * 8)


# ---- reading the answer -----------------------------------------------------------------------------------------------
FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)


def parse_calls(text, tools):
    """The model's answer -> (calls, kind). kind is "calls" (calls is the list in OpenAI shape), "text" (a normal answer) or
    "broken" (it tried to call a tool and did not get it right)."""
    t = (text or "").strip()
    m = FENCE.match(t)
    if m:
        t = m.group(1)
    if not t.startswith("{"):
        return None, "text"
    try:
        obj = json.loads(t)
    except ValueError:
        return None, "broken" if '"tool_calls"' in t or '"arguments"' in t else "text"
    items = obj.get("tool_calls") if isinstance(obj, dict) and "tool_calls" in obj else ([obj] if isinstance(obj, dict) and "arguments" in obj and "name" in obj else None)
    if not isinstance(items, list) or not items:
        return None, "text"
    by_name = {x["name"]: x for x in tools}
    calls = []
    for it in items:
        if not isinstance(it, dict) or it.get("name") not in by_name:
            return None, "broken"
        args = it.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                return None, "broken"
        text_args = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
        if not constrain.check(by_name[it["name"]]["parameters"], text_args):
            return None, "broken"
        calls.append({"id": "call_" + uuid.uuid4().hex[:16], "type": "function", "function": {"name": it["name"], "arguments": text_args}})
    return calls, "calls"
