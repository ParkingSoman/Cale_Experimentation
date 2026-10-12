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
| `max_tokens`, `max_completion_tokens` | 1 to 4096. Default 1024. |
| `temperature` | 0 to 2. Default 0. |
| `cale_deadline_s`, `cale_allow_disk`, `cale_on_miss` | goals, only with `gemma-3n:auto` |

Fields not listed here are not supported.

Reply headers: `X-Cale-Size`, `X-Cale-Predicted-S`, `X-Cale-Goal-Met`, `X-Cale-Disk-MB`.

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

All errors use `{"error": {"message", "type", "code"}}`. Status codes: 400 bad request, 401 bad key,
403 not allowed, 404 unknown model, 413 too large, 422 goal missed (with `on_miss: "refuse"`), 429 busy.
