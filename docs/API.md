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
| `max_tokens`, `max_completion_tokens` | at least 1. The prompt plus this number must fit the context (see below). Default 1024, or less if the prompt is long. |
| `temperature` | 0 to 2. Default 0. |
| `top_p` | 0 to 1. Default 1. |
| `seed` | an integer. The same seed, prompt and settings give the same answer. Without a seed, each call differs when `temperature` is above 0. |
| `stop` | a string, or a list of up to 4 strings. The answer ends before the first match. `finish_reason` is `stop`. |
| `cale_deadline_s`, `cale_allow_disk`, `cale_on_miss` | goals, only with `gemma-3n:auto` |
| `user`, `stream_options`, `store`, `metadata` | accepted. They change nothing. |

Cale never ignores a field in silence.

| You send | Cale answers |
|---|---|
| `tools`, `tool_choice`, `response_format` (other than `text`), `n` above 1, penalties other than 0, images | `400 unsupported_parameter`. The `param` field names it. These are planned, not available yet. |
| a field Cale has never heard of | `400 unknown_parameter` |
| a bad value | `400 invalid_...` with the field name in `param` |

### Context limit

Each computer has a limit for prompt tokens plus answer tokens. `GET /v1/models` shows it as `cale.max_context_tokens`.
It can differ between computers. A request over the limit gets `400 context_length_exceeded`. The reply lists
`prompt_tokens`, `max_tokens` and `max_context_tokens`. If you send no `max_tokens`, Cale shortens the default to fit.

### What the reply says about speed and goals

| Header or field | Meaning |
|---|---|
| `X-Cale-Size` | the size that ran |
| `X-Cale-Waited-S`, `cale.waited_s` | seconds the request waited for other apps |
| `X-Cale-Predicted-S` | the prediction, including the wait so far |
| `X-Cale-Goal-Met` | in a reply that is not streamed: whether the deadline was met in the end, wait included |
| `cale.goal_met_actual`, `cale.total_s` | the same, in the body (also in the last chunk of a stream) |
| `X-Cale-Memory-Goal-Met` | a model object with `memory_pct`: whether a size fit the memory goal |

### A client that leaves

If your app closes the connection while Cale is working (reading a long prompt, or writing the answer), Cale stops that work
at once. Other apps do not wait for an answer nobody will read. A request that is still waiting in line is dropped.

### Engine failure

If the engine fails during a call, you get `500 engine_error`. Cale unloads the model and loads it again on the next call.
`GET /cale/status` shows `last_error`.

## 4. Model objects (goals)

A model object holds what your app needs. Cale returns an id. Use that id as `model` in chat calls.

`POST /cale/models` with a JSON body. The reply stays open. The object lives while the connection is open.

| Field | Meaning |
|---|---|
| `name` | becomes the id |
| `size` | exact size: `"smart"`, `"fast"` or a percent. Not with goals. |
| `deadline_s` | finish one answer in this many seconds. Needs `max_tokens`. |
| `max_tokens` | answer length for the deadline, and the default limit |
| `memory_pct` | keep Cale under this share of the computer's RAM |
| `allow_disk` | allow reading missing weights from disk. Default false. |
| `on_miss` | `"smallest"` (default) or `"refuse"` |

The first line of the reply is JSON with `id`. After that Cale sends a blank line every second.
Python apps use `cale.client.model(base_url, key, ...)`.

## 5. Holds

`POST /cale/load` with `{"size": ...}` or `{"memory_pct": n}` keeps a size in RAM while the connection stays open.
`POST /cale/unload` with `{"hold": id}` ends it early.

## 6. List models

`GET /v1/models` (with key) lists sizes, your model objects, and what Cale measured on this computer.

## Errors

All errors use `{"error": {"message", "type", "code", "param"}}`. Status codes: 400 bad request (including
`context_length_exceeded`), 401 bad key, 403 not allowed, 404 unknown model, 413 body too large, 422 goal missed
(with `on_miss: "refuse"`), 429 busy, 500 `engine_error`.
