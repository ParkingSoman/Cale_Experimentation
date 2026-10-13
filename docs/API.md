# Cale HTTP API

Cale listens on `http://127.0.0.1:47821`. Use this exact host, not `localhost`.
Call it from an app's own backend. Cale refuses any request that carries an `Origin` header.
Send `Content-Length`. Chunked request bodies are refused.

## 1. Check that Cale runs

| Request | Reply |
|---|---|
| `GET /cale/status` (no key) | JSON with `ok`, `loaded`, `pressure`, `holds` |

Python apps can start Cale if it is not running:

```python
from cale.client import ensure_running
ensure_running()
```

## 2. Get a key

`POST /cale/connect` with `{"app": "My App"}`. The computer shows an Allow question to the person.
The call waits for the answer (up to 2 minutes).

| Result | Reply |
|---|---|
| Allowed | `200 {"app", "api_key", "base_url"}` |
| Not allowed | `403 {"error": {"code": "denied"}}` |
| Another app is waiting for the person | `429 busy` |

Send the key as `Authorization: Bearer KEY` on every other call. Keep the key. Do not ask for a new one each time.

## 3. Chat

`POST /v1/chat/completions`, OpenAI style. `stream: true` is supported.

| Field | Support |
|---|---|
| `model` | `gemma-3n:smart`, `gemma-3n:fast`, `gemma-3n:<48-100>`, `gemma-3n:auto`, or the id of a model object |
| `messages` | roles `system`, `user`, `assistant`, `developer`. Text only. |
| `max_tokens`, `max_completion_tokens` | at least 1. The prompt plus this number must fit the context (see below). With no limit sent, only the context limit applies: an answer ends when the model stops or the context is full. |
| `temperature` | 0 to 2. Default 0. |
| `top_p` | 0 to 1. Default 1. |
| `seed` | an integer. The same seed, prompt, settings and size give the same answer. A seeded call does not use the prompt cache. If other answers run in the same step, tiny differences in the arithmetic can change a sampled answer in rare cases. Without a seed, each call differs when `temperature` is above 0. |
| `stop` | a string, or a list of up to 4 strings. The answer ends before the first match. `finish_reason` is `stop`. |
| `tools`, `tool_choice`, `parallel_tool_calls` | function tools, as in the OpenAI API (see "Tools" below) |
| `response_format` | `text`, `json_object` or `json_schema` (see "Structured output" below) |
| `cale_min_tokens_per_s`, `cale_max_first_token_s`, `cale_memory_pct`, `cale_min_quality`, `cale_power`, `cale_priority`, `cale_on_miss`, `cale_max_wait_s`, `cale_allow_disk` | goals, only with `gemma-3n:auto` (see section 4) |
| `user`, `stream_options`, `store`, `metadata` | accepted. They change nothing. |

Cale never ignores a field in silence.

| You send | Cale answers |
|---|---|
| `functions`, `function_call`, `logprobs`, `n` above 1, penalties other than 0, images, audio | `400 unsupported_parameter`. The `param` field names it. These are planned, not available yet. |
| a field Cale has never heard of | `400 unknown_parameter` |
| a bad value | `400 invalid_...` with the field name in `param` |

### Structured output

| `response_format` | What you get |
|---|---|
| `{"type": "text"}` or none | plain text |
| `{"type": "json_object"}` | one JSON object |
| `{"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}` | one JSON value that fits the schema |

This is not a request to the model. Cale checks every token as it is chosen, so the answer is always valid JSON that fits the schema,
on every computer and at every size. If `max_tokens` cuts the answer short, `finish_reason` is `length` and the JSON is incomplete.
Supported schema keywords: `type` (also a list), `properties`, `required`, `additionalProperties`, `items`, `minItems`, `maxItems`,
`enum`, `const`, `anyOf`, `oneOf`, `minLength`, `maxLength`. An object with `properties` takes only those keys unless
`additionalProperties` is `true`. At most 2 blank characters between tokens. A schema Cale cannot use gets `400 invalid_schema`.
A very strict schema that the model's tokens cannot spell gets `422 constraint_dead_end`.

### Tools

`tools` and `tool_choice` work as in the OpenAI API. Gemma 3n has no built-in tool calling: Cale shows the tools to the model in the prompt and
checks the answer.

| `tool_choice` | What happens |
|---|---|
| `"auto"` (default with `tools`) | The model decides. A tool call comes back as `tool_calls` with `finish_reason: "tool_calls"`. If the model tried to call a tool and got it wrong, Cale asks again with the answer guided to a valid call. |
| `"required"` | Always a valid tool call, whatever the user wrote |
| `{"type": "function", "function": {"name": "..."}}` | A valid call to that tool |
| `"none"` | Tools are not shown to the model |

Every `arguments` string fits the tool's `parameters` schema. `parallel_tool_calls: false` allows one call. Send the result back as a
`tool` message with the `tool_call_id`; an assistant message may carry its `tool_calls`. A streamed answer with tools arrives at the end
of the work (Cale must read the whole answer to know if it is a tool call), in OpenAI's chunk shape.
You cannot send `tools` and `response_format` in one call yet.

### Prompt cache

A chat that grows by a turn does not read its history again. `usage.prompt_tokens_details.cached_tokens` says how many prompt tokens were reused.
It works when the new prompt starts with the previous prompt and the size is the same. One conversation is kept at a time.

### Context limit

Each computer has a limit for prompt tokens plus answer tokens. `GET /v1/models` shows it as `cale.max_context_tokens`.
It can differ between computers. A request over the limit gets `400 context_length_exceeded`. The reply lists
`prompt_tokens`, `max_tokens` and `max_context_tokens`. If you send no `max_tokens`, Cale shortens the limit to fit.

### What the reply says about speed and goals

| Header or field | Meaning |
|---|---|
| `X-Cale-Size` | the size that ran (it may change during the answer: see `cale.size_changes`) |
| `X-Cale-Waited-S`, `cale.waited_s` | seconds the request waited for other apps |
| `X-Cale-Predicted-Tokens-Per-S`, `X-Cale-Predicted-First-Token-S` | what Cale predicted for the size it chose |
| `X-Cale-Confidence` | `estimate` (Cale has seen no answer on this computer yet) or `learned` |
| `X-Cale-Goal-Met` | only when goals were given. In a reply that is not streamed: whether the goals were met in the end, wait included |
| `X-Cale-Unmet` | the goals that were not met (`speed`, `first_token`, `memory`, `quality`) |
| `cale.goals_actual` | for each goal: the goal, what really happened, `met` |
| `cale.goal_met_actual`, `cale.total_s` | the same in one value, and the total seconds (also in the last chunk of a stream) |
| `cale.size_changes` | sizes Cale changed to during the answer to keep a speed goal |
| `cale.answer_limit` | the limit on answer tokens that applied |
| `X-Cale-Memory-Goal-Met` | a model object with `memory_pct`: whether a size fit the memory goal |

### A client that leaves

If your app closes the connection while Cale is working (reading a long prompt, or writing the answer), Cale stops that work
at once. Other apps do not wait for an answer nobody will read. A request that is still waiting in line is dropped.

### Engine failure

If the engine fails during a call, you get `500 engine_error`. Cale unloads the model and loads it again on the next call.
`GET /cale/status` shows `last_error`.

## 4. Model objects and goals

A model object holds what your app needs. Cale returns an id. Use that id as `model` in chat calls. With `gemma-3n:auto`
you can send the same goals on one call, with a `cale_` prefix.

`POST /cale/models` with a JSON body. The reply stays open. The object lives while the connection is open.

| Field | Meaning |
|---|---|
| `name` | becomes the id |
| `size` | exact size: `"smart"`, `"fast"` or a percent. Not with goals. |
| `min_tokens_per_s` | answer at least this fast. Kept during the answer by changing the size between tokens. |
| `max_first_token_s` | first token within this many seconds of the call arriving (waiting in line and reading the prompt count) |
| `memory_pct` | keep Cale under this share of the computer's RAM, counting weights and context |
| `min_quality` | 0 to 1. No size that repeats less than this share of the full size's next-word choices. |
| `power` | `"normal"` or `"light"` (a smaller size on battery or when the computer is hot) |
| `priority` | list of `"memory"`, `"quality"`, `"first_token"`, `"speed"`: which goal to keep when they conflict, most important first |
| `on_miss` | when no size meets the goals: `"best_effort"` (default), `"refuse"`, `"wait"` or `"try"`. `"smallest"` is the old name of `"best_effort"`. |
| `max_wait_s` | with `"wait"`: how long to keep trying, 0 to 60 s. Default 30. While Cale waits, other apps wait too. |
| `allow_disk` | allow reading missing weights from disk. Default false. |
| `max_tokens` | the length limit of this object's answers |

`deadline_s` was removed (400 `unsupported_parameter`). Cale cannot know how long an answer will be.

The first line of the reply is JSON with `id` and `prediction` (see `/cale/predict`). After that Cale sends a blank line every second.
With `on_miss: "refuse"` an object whose goals this computer cannot meet is refused at once with 422.
Python apps use `cale.client.model(base_url, key, ...)`.

### `POST /cale/predict`: ask before you run

Same goal fields, plus `prompt_tokens` (default 200) and `max_tokens` (default 256). Nothing runs and it never refuses a goal.

| Field of the reply | Meaning |
|---|---|
| `goal_met`, `unmet` | whether the goals can be met, and which cannot |
| `chosen` | `size`, `size_pct`, `tokens_per_s`, `first_token_s`, `memory_gb`, `quality` of the size Cale would use |
| `sizes` | the same numbers for every size, and whether each is `safe` |
| `floor` | `fastest` and `smallest`: the most this computer can do |
| `confidence` | `estimate` or `learned` |
| `unsafe` | true if no size can run this call safely |

Python apps use `cale.client.predict(base_url, key, ...)`.

### The safety limit

Cale never runs what it predicts would freeze or crash the computer, whatever `on_miss` says, and for exact sizes too.
The reply is `507 unsafe_for_this_computer`. `error.cale` has `needed_gb`, `ram_gb`, `available_gb` and the `smallest_size`.

## 5. Holds

`POST /cale/load` with `{"size": ...}` or `{"memory_pct": n}` keeps a size in RAM while the connection stays open.
`POST /cale/unload` with `{"hold": id}` ends it early.

## 6. List models

`GET /v1/models` (with key) lists sizes, your model objects, and what Cale predicts for this computer (`cale.this_computer`: speed and memory of every size, `floor`, `confidence`).

## Errors

All errors use `{"error": {"message", "type", "code", "param"}}`. Status codes: 400 bad request (including
`context_length_exceeded`), 401 bad key, 403 not allowed, 404 unknown model, 413 body too large, 422 goal missed
(with `on_miss: "refuse"` or `"wait"`), 429 busy, 500 `engine_error`, 507 `unsafe_for_this_computer`.
