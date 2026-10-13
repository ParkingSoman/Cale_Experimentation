"""Structured output and tools against a running Cale (start one with scripts/cale_dev_server.py HOME).
JSON that always fits a schema, tool calls, tool results in the conversation, streaming with tools.

usage: python cale/tests/structured.py [--base http://127.0.0.1:47899]
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cale import constrain  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:47899")
a = ap.parse_args()
FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:200]}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


def call(path, body=None, key=None, stream=False):
    h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    req = urllib.request.Request(a.base + path, data=None if body is None else json.dumps(body).encode(), headers=h)
    try:
        r = urllib.request.urlopen(req, timeout=900)
        if stream:
            return r.status, [json.loads(l[6:]) for l in r.read().decode().splitlines() if l.startswith("data: ") and l != "data: [DONE]"]
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def chat(key, content, **kw):
    msgs = content if isinstance(content, list) else [{"role": "user", "content": content}]
    return call("/v1/chat/completions", {"model": "gemma-3n:smart", "messages": msgs, **kw}, key, stream=bool(kw.get("stream")))


def chunks_of(r):
    """The parsed chunks of a streamed reply, or [] (and a message) if the call failed with a JSON error."""
    return r if isinstance(r, list) else []


key = call("/cale/connect", {"app": "Structured Test"})[1]["api_key"]

# 1. JSON that fits a schema
person = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "city": {"type": "string"},
                                              "role": {"enum": ["student", "teacher", "other"]}}, "required": ["name", "age", "city", "role"]}
fmt = {"type": "json_schema", "json_schema": {"name": "person", "schema": person, "strict": True}}
for text in ("Extract the person: Maria Lopez is a 34 year old teacher who lives in Madrid.",
             "Extract the person: a student called Wei, 19, studies in Singapore.",
             "Write anything you like about dogs."):
    s, r = chat(key, text, response_format=fmt, max_tokens=120)
    out = r["choices"][0]["message"]["content"] if s == 200 else ""
    check(f"json_schema: valid and fits ({text[:30]}...)", s == 200 and constrain.check(person, out) and r["choices"][0]["finish_reason"] == "stop", (s, out))
s, r = chat(key, "Extract the person: Maria Lopez is a 34 year old teacher who lives in Madrid.", response_format=fmt, max_tokens=120, temperature=0)
d = json.loads(r["choices"][0]["message"]["content"])
check("json_schema: the content is right too", d["name"] == "Maria Lopez" and d["age"] == 34 and d["city"] == "Madrid" and d["role"] == "teacher", d)
s, r = chat(key, "Give me two colors.", response_format={"type": "json_object"}, max_tokens=100)
out = r["choices"][0]["message"]["content"]
check("json_object: a JSON object", s == 200 and isinstance(json.loads(out), dict), out)
s, r = chat(key, "Pick a number.", response_format={"type": "json_schema", "json_schema": {"schema": {"type": "integer"}}}, max_tokens=20)
check("a bare integer schema gives an integer", s == 200 and isinstance(json.loads(r["choices"][0]["message"]["content"]), int), r["choices"][0]["message"]["content"])
s, r = chat(key, "Say hello in JSON.", response_format={"type": "json_object"}, max_tokens=3)
check("cut off by max_tokens: finish_reason is length", s == 200 and r["choices"][0]["finish_reason"] == "length", r["choices"][0])
s, r = chat(key, "x", response_format={"type": "json_schema", "json_schema": {"schema": {"type": "frog"}}})
check("a schema Cale cannot use: 400 invalid_schema", s == 400 and r["error"]["code"] == "invalid_schema", r)
s, r = chat(key, "x", response_format={"type": "xml"})
check("unknown format type: 400", s == 400 and r["error"]["code"] == "invalid_response_format", r)
s, chunks = chat(key, "Extract the person: Maria Lopez is a 34 year old teacher who lives in Madrid.", response_format=fmt, max_tokens=120, stream=True)
streamed = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks_of(chunks))
check("json_schema streams, and the pieces add up to valid JSON", constrain.check(person, streamed), (s, chunks if not isinstance(chunks, list) else streamed))

# 2. tools
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather in a city",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"enum": ["c", "f"]}}, "required": ["city"]}}},
         {"type": "function", "function": {"name": "add", "description": "Add two numbers", "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}}}]
s, r = chat(key, "What is the weather in Paris right now?", tools=TOOLS, max_tokens=120)
m, fr = r["choices"][0]["message"], r["choices"][0]["finish_reason"]
call1 = (m.get("tool_calls") or [{}])[0]
check("tools auto: calls the right tool with valid arguments", s == 200 and fr == "tool_calls" and m["content"] is None and call1["function"]["name"] == "get_weather"
      and json.loads(call1["function"]["arguments"])["city"].lower() == "paris" and call1["id"] and call1["type"] == "function", m)
s, r = chat(key, "Tell me one short fact about the moon.", tools=TOOLS, max_tokens=80)
m = r["choices"][0]["message"]
check("tools auto: a question that needs no tool gets a text answer", s == 200 and m.get("content") and not m.get("tool_calls") and r["choices"][0]["finish_reason"] in ("stop", "length"), m)
s, r = chat(key, "What is the weather in Paris?", tools=TOOLS, tool_choice="none", max_tokens=60)
check("tool_choice none: no tool calls", s == 200 and not r["choices"][0]["message"].get("tool_calls"), r["choices"][0]["message"])
s, r = chat(key, "Hello there.", tools=TOOLS, tool_choice="required", max_tokens=120)
m = r["choices"][0]["message"]
check("tool_choice required: always a valid tool call", s == 200 and m.get("tool_calls") and all(constrain.check(next(t["function"]["parameters"] for t in TOOLS if t["function"]["name"] == c["function"]["name"]), c["function"]["arguments"]) for c in m["tool_calls"]), m)
s, r = chat(key, "Hello there.", tools=TOOLS, tool_choice={"type": "function", "function": {"name": "add"}}, max_tokens=120)
m = r["choices"][0]["message"]
check("tool_choice naming a tool: that tool", s == 200 and [c["function"]["name"] for c in m["tool_calls"]] == ["add"], m)
s, r = chat(key, "What is 12 plus 30? Also, what is the weather in Rome?", tools=TOOLS, tool_choice="required", parallel_tool_calls=False, max_tokens=200)
check("parallel_tool_calls false: one call", s == 200 and len(r["choices"][0]["message"]["tool_calls"]) == 1, r["choices"][0]["message"])
s, chunks = chat(key, "What is the weather in Paris right now?", tools=TOOLS, max_tokens=120, stream=True)
tcs = [t for c in chunks_of(chunks) for t in (c["choices"][0]["delta"].get("tool_calls") or [])]
args = "".join(t["function"].get("arguments", "") for t in tcs)
check("tools stream: a tool call in OpenAI's chunk shape", tcs and tcs[0]["function"]["name"] == "get_weather" and json.loads(args)["city"].lower() == "paris"
      and chunks_of(chunks)[-1]["choices"][0]["finish_reason"] == "tool_calls", chunks)

# 3. the whole loop: the model calls, the app answers with a tool message, the model uses the result
s, r = chat(key, "What is 12 plus 30?", tools=TOOLS, tool_choice="required", max_tokens=120)
tc = r["choices"][0]["message"]["tool_calls"][0]
history = [{"role": "user", "content": "What is 12 plus 30?"}, {"role": "assistant", "content": None, "tool_calls": [tc]},
           {"role": "tool", "tool_call_id": tc["id"], "content": "42"}]
s, r = chat(key, history, tools=TOOLS, max_tokens=80)
final = r["choices"][0]["message"]
check("after a tool result the model answers with it", s == 200 and "42" in (final.get("content") or "") and not final.get("tool_calls"), final)

# 4. refusals
s, r = chat(key, "x", tools=TOOLS, response_format={"type": "json_object"})
check("tools with response_format: 400", s == 400 and r["error"]["code"] == "unsupported_parameter", r)
s, r = chat(key, "x", tools=TOOLS, tool_choice={"type": "function", "function": {"name": "nope"}})
check("tool_choice naming an unknown tool: 400", s == 400 and r["error"]["code"] == "invalid_tool_choice", r)
s, r = chat(key, "x", tools=[TOOLS[0], TOOLS[0]])
check("two tools with one name: 400", s == 400 and r["error"]["code"] == "invalid_tools", r)
s, r = chat(key, "x", tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "frog"}}}])
check("a tool with a schema Cale cannot use: 400", s == 400 and r["error"]["code"] == "invalid_tools", r)
s, r = chat(key, "x", tools=TOOLS, parallel_tool_calls="yes")
check("parallel_tool_calls must be true or false", s == 400 and r["error"]["code"] == "invalid_parallel_tool_calls", r)

print("\nFAILED: " + ", ".join(FAIL) if FAIL else "\nall passed")
sys.exit(1 if FAIL else 0)
