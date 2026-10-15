# Cale HTTP API

Cale listens on `http://127.0.0.1:47821`. Use this exact host, not `localhost`.
Call it from an app's own backend. Cale refuses any request that carries an `Origin` header.
Send `Content-Length`. Chunked request bodies are refused.

## 1. Check that Cale runs

| Request | Reply |
|---|---|
| `GET /cale/status` (no key) | JSON with `ok`, `loaded`, `slots`, `pressure`, `holds` |
| `GET /cale/computer` (no key) | what this computer can do, as last learned: predicted speed and memory of every size, the floor, and `max_context_tokens` (the limit for prompt plus answer tokens). It never loads the model. For a full answer ask `POST /cale/predict` (with a key) or run `cale computer`. |
| `GET /cale/activity` (Cale Hub key only) | the calls that run now (app, size, pieces so far, first-token time) and the last 50 finished calls (app, size, tokens, seconds, tokens/s, first-token time, wait, goal met). Numbers only: never prompt or answer text. Any other key gets `403 hub_only`. The app name "Cale Hub" is reserved: `POST /cale/connect` refuses it. |

Start Cale if it is not running, from any language: run `cale start --wait` (a command installed with Cale). Python apps can also call
`cale.client.ensure_running()`, Node apps `ensureRunning()` from `cale/client.mjs`.

## 2. Get a key

`POST /cale/connect` with `{"app": "My App"}`. The person at the computer must allow the app. With a screen they see a question.
Without one (a server, an SSH session), the request waits until the person types `cale allow "My App"` in a terminal on that computer.

| Body field | Meaning |
|---|---|
| `app` | the app's name, shown to the person |
| `wait_s` | how long this call may wait for the answer: 0 to 600, default 120. `0` answers at once with a ticket. |

| Result | Reply |
|---|---|
| Allowed | `200 {"app", "api_key", "base_url"}` |
| Not allowed, or no answer in time | `403 {"error": {"code": "denied"}}` |
| Not answered yet | `202 {"status": "pending", "ticket", "poll", "message"}`. The message says what the person must do. |
| Another app is waiting for the person (screen only) | `429 busy` |
| An app with the same name is already waiting | `409 already_waiting` (it must not share the other request's ticket) |
| Too many requests are waiting | `429 busy` |

An app whose HTTP client cannot wait minutes uses `wait_s: 0` and polls `GET /cale/connect/status?ticket=...` every 2 s: `202` while pending,
`200` with the key once (the key is handed out one time), `403` if denied, `404` for an unknown ticket.

The person's side, without a screen:

| Command | Does |
|---|---|
| `cale pending` | lists the apps that wait |
| `cale allow NAME` | allows one. Must be typed in a terminal. It shows the waiting app (exact name and ticket) and asks "Allow ...? [y/N]". If two requests have the same name, it asks for the ticket. |
| `cale deny NAME` | says no |

Cale never opens a network door for this: the command and Cale talk through Cale's private folder, as keys do. Anyone who can write there
can already read the keys.

Send the key as `Authorization: Bearer KEY` on every other call. Keep the key. Do not ask for a new one each time.

| Rule | Detail |
|---|---|
| One key for each app name | A second `POST /cale/connect` for the same app name makes a new key. The old key stops at once: calls with it get `401 invalid_api_key`. An app must store its key (a file, or its settings) and reuse it. It asks again only after a `401`. |
| `202` is not an error | `202 pending` is a normal success status. An HTTP library that raises on anything but `200` must be told to accept `202`, or use `wait_s` long enough, or poll with a client that reads the body. |
| Reserved name | The app name "Cale Hub" belongs to Cale. `POST /cale/connect` with it gets `400 reserved_name`. |

## 3. Chat

`POST /v1/chat/completions`, OpenAI style. `stream: true` is supported.

| Field | Support |
|---|---|
| `model` | `gemma-3n:smart`, `gemma-3n:fast`, `gemma-3n:<48-100>`, `gemma-3n:auto`, or the id of a model object |
| `messages` | roles `system`, `user`, `assistant`, `developer`, and `tool` (a tool's result, see Tools). Content: text, and pictures where `capabilities.images` is true (see Pictures). |
| `max_tokens`, `max_completion_tokens` | at least 1. The prompt plus this number must fit the context (see below). With no limit sent, only the context limit applies: an answer ends when the model stops or the context is full. |
| `temperature` | 0 to 2. Default 0. |
| `top_p` | 0 to 1. Default 1. |
| `seed` | an integer. The same seed, prompt, settings and size give the same answer. A seeded call does not use the prompt cache. If other answers run in the same step, tiny differences in the arithmetic can change a sampled answer in rare cases. Without a seed, each call differs when `temperature` is above 0. |
| `stop` | a string, or a list of up to 4 strings. The answer ends before the first match. `finish_reason` is `stop`. |
| `tools`, `tool_choice`, `parallel_tool_calls` | function tools, as in the OpenAI API (see "Tools" below). At most 32 tools. |
| `response_format` | `text`, `json_object` or `json_schema` (see "Structured output" below) |
| `cale_min_tokens_per_s`, `cale_max_first_token_s`, `cale_memory_pct`, `cale_min_quality`, `cale_power`, `cale_priority`, `cale_on_miss`, `cale_max_wait_s`, `cale_allow_disk` | goals, only with `gemma-3n:auto` (see section 4) |
| `user`, `stream_options`, `store`, `metadata` | accepted. They change nothing. |

Cale never ignores a field in silence.

| You send | Cale answers |
|---|---|
| `functions`, `function_call`, `n` above 1, penalties other than 0 | `400 unsupported_parameter`. The `param` field names it. These are planned, not available yet. |
| sound (`input_audio`) on a computer whose engine cannot read sound | `400 unsupported_content`. Ask first: `capabilities.audio` in `GET /v1/models` |
| a picture on a computer whose engine cannot read pictures | `400 unsupported_content`. Ask first: `capabilities.images` in `GET /v1/models` |
| a field Cale has never heard of | `400 unknown_parameter` |
| a bad value | `400 invalid_...` with the field name in `param` |

### Sound

Send one WAV clip in a user message:

```json
{"role": "user", "content": [
  {"type": "text", "text": "Transcribe this. Reply with the transcript only."},
  {"type": "input_audio", "input_audio": {"data": "UklGR...", "format": "wav"}}]}
```

| Rule | Value |
|---|---|
| Where | user messages only |
| Format | WAV only, base64 in `data`, `format` is `wav`. Any sample rate (Cale converts to 16 kHz mono). |
| Limits | 1 clip per request, 30 seconds and 10 MB at most |
| Cost | about 6 prompt tokens for each second of sound |
| Errors | `400 invalid_audio` (bad data or over a limit), `400 unsupported_content` (this engine cannot read sound) |
| Where it works | Mac now. Windows and Linux: not yet. Check `capabilities.audio` in `GET /v1/models`. |

### Known differences between computers

| Difference | What to do |
|---|---|
| On Windows and Linux, an answer may end with `finish_reason: "length"` a little before `max_tokens` when other apps fill the shared context | Treat `length` as "more may exist" and ask again if you need the rest |
| The memory goal is measured as GPU memory on a Mac and as process memory on Windows and Linux | Read `goals_actual.memory` for what was measured |
| On Windows and Linux, the first answer to a prompt and a later answer that reuses the prompt cache can differ at `temperature` 0, because the arithmetic is split differently (tiny rounding). The Mac gives the same text | Send a `seed`. A seeded call does not use the prompt cache, so it repeats exactly |
| On a computer where a GPU reads long prompts, an app that leaves in the middle of a long prompt can hold the next app up for about 6 seconds (one piece of the prompt cannot be stopped). Without that GPU, it is about 1 to 2 seconds | None needed. `GET /cale/status` shows `placement` |
| After an OS update the graphics driver may differ from the one Cale was measured with. Cale then does not use its saved choice and may run on the CPU, which can be slower | `GET /cale/status` shows `placement` and `placement_stale` (`true` when the saved choice no longer fits this computer). `cale tune` measures again |
| A graphics card that is nearly full can fail while Cale loads the model. Cale tries the card in a child process first, uses the CPU if that fails, and starts again on the CPU any answer that had not said a word. For about 10 minutes the CPU is used, then the card is tried again | Your app sees a normal answer, maybe a few seconds late. The log says why |
| Pictures and sound work on a Mac only | Check `capabilities.images` and `capabilities.audio` first |

### Log-probabilities

Send `"logprobs": true` and, if you want the runners-up, `"top_logprobs": 0 to 20`. Each choice then has
`logprobs.content`: one entry per answer token, as in the OpenAI API.

```json
{"token": " red", "logprob": -0.21, "bytes": [32, 114, 101, 100],
 "top_logprobs": [{"token": " red", "logprob": -0.21, "bytes": [...]}, {"token": " blue", "logprob": -1.9, "bytes": [...]}]}
```

| Rule | Value |
|---|---|
| What the number is | the natural log of the probability the model gave that token, over the whole vocabulary, before `temperature` and `top_p` |
| Streaming | each chunk carries the entries of the tokens that came with it |
| Not together with | `tools` or `response_format` (`400 unsupported_parameter`) |
| Errors | `400 invalid_logprobs`, `400 invalid_top_logprobs` (not 0 to 20, or `logprobs` is not true) |
| Cost | one more pass over the vocabulary for each token. Only calls that ask for it pay. |

### Pictures

Send a picture in a user message, as in the OpenAI API:

```json
{"role": "user", "content": [
  {"type": "text", "text": "What is in this picture?"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}]}
```

| Rule | Value |
|---|---|
| Where | user messages only |
| How | a `data:image/...;base64,...` URL only. Cale never fetches a web address. |
| Limits | 8 pictures per request, 10 MB and 40 million pixels each |
| Cost | each picture is 256 tokens of the prompt. They count toward the context limit. |
| Errors | `400 invalid_image` (bad data or over a limit), `400 unsupported_content` (this engine cannot read pictures) |
| Prompt cache | a prompt with a picture is read whole each time |
| Where it works | Mac now. Windows and Linux: not yet. Check `capabilities.images` in `GET /v1/models` before you send a picture. |

The part of the model that reads pictures loads for the call and is freed after it. A text call never pays for it.

### Structured output

| `response_format` | What you get |
|---|---|
| `{"type": "text"}` or none | plain text |
| `{"type": "json_object"}` | one JSON object |
| `{"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}` | one JSON value that fits the schema |

This is not a request to the model. Cale checks every token as it is chosen, so the answer is always valid JSON that fits the schema,
on every computer and at every size. If `max_tokens` cuts the answer short, `finish_reason` is `length` and the JSON is incomplete.
Supported schema keywords: `type` (also a list), `properties`, `required`, `additionalProperties`, `items`, `minItems`, `maxItems`,
`enum`, `const`, `anyOf`, `oneOf`, `minLength`, `maxLength`. Any other keyword is ignored without an error, so it is not enforced. This includes `pattern`, `format`, `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `multipleOf`, `uniqueItems`, `minProperties`, `maxProperties`, `patternProperties`, `prefixItems`, `allOf`, `not`, `if`, `$ref` and `default`. Check those in your own code after the answer. A type Cale does not know gets `400 invalid_schema`. An object with `properties` takes only those keys unless
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
It is in a reply that is not streamed, and in the last chunk of a stream (not in a stream with `tools` yet). `0` means nothing was reused.

| Rule | Detail |
|---|---|
| What is kept | the previous prompt of the app |
| What is reused | only the start of the new prompt that has exactly the same tokens as the kept prompt. One changed word early in the prompt ends the reuse at that word. |
| Same size | the cache is kept per size. A call at another size starts again. |
| Same app | only the app that made a conversation can reuse it. One app never gains speed from another app's prompt. |
| How many | Cale keeps only a few conversations (one on a Mac), so another call can push yours out. |
| No cache | a call with `seed`, and a call with pictures or sound |

What this means for a long document:

| Do | Why |
|---|---|
| Put the document FIRST (in the `system` message or the first `user` message). Keep it identical, character for character. | Only a same start is reused. |
| Put the question LAST. | The question changes. The start must not. |
| Do not put a date, a name or an id before the document. | A change before the document ends the reuse. |

Windows and Linux (the llama.cpp engine) reuse the common start of the new prompt and the kept prompt, if it is at least 16 tokens. So a different question after the same document still reuses the document. A Mac (the MLX engine) reuses the kept prompt only when the whole kept prompt is the start of the new prompt, and the new prompt is longer. On a Mac, the best case is a chat that grows by a turn. A new question after the same document reuses the document only if the kept prompt was the document alone. Test with `cached_tokens` on the computers you care about.

### Context limit

Each computer has a limit for prompt tokens plus answer tokens. `GET /v1/models` shows it as `cale.max_context_tokens`.
`GET /cale/computer` (no key) shows it too, as `max_context_tokens`. It can differ between computers. A request over the limit gets
`400 context_length_exceeded`, with the numbers in `error.cale`:

```json
{"error": {"message": "This request needs 40012 prompt tokens plus 256 answer tokens. This computer accepts 32768 in total.",
           "type": "invalid_request_error", "code": "context_length_exceeded", "param": "messages",
           "cale": {"prompt_tokens": 40012, "max_tokens": 256, "max_context_tokens": 32768}}}
```

If you send no `max_tokens`, Cale shortens the limit to fit. Then `max_tokens` in the error is `32768`, and the call fails only when the prompt alone does not fit.

Cale has no endpoint that counts tokens. To estimate: English text is about 4 characters for each token (other languages and code use more tokens per character). A picture is 256 tokens. Sound is about 6 tokens for each second. For the exact count, make one chat call with `max_tokens: 1` and read `usage.prompt_tokens`. `POST /cale/predict` takes a number, not text.
`POST /cale/predict` takes `prompt_tokens` (your estimate) and says what Cale would do for a prompt of that size. It does not include the time to load the model or the time to wait for other apps: the numbers are for a model that is loaded and an empty line. A call that loads the model first counts that time in `waited_s` and `first_token` (see "Cold start" in section 4).

### A prompt with no user message

Gemma puts the text of a `system` (or `developer`) message into the first `user` turn. A prompt with only a `system` message has no user turn, so the system text is not sent: the prompt is only a few tokens (a developer saw 4), and the answer does not use your text. Always send at least one `user` message. Put your instructions in `system` and the work in `user`.

### What the reply says about speed and goals

| Header or field | Meaning |
|---|---|
| `X-Cale-Size` | the size that ran, as a percent of the full model: `100` is the full size, `48` the smallest trained size. The same number is in `gemma-3n:<percent>`. It may change during the answer: see `cale.size_changes`. In the reply body, `model` is `gemma-3n` for 100 and `gemma-3n:<percent>` for the others. With a model object, `model` is the object's id. |
| `X-Cale-Waited-S`, `cale.waited_s` | seconds the request waited for other apps |
| `X-Cale-Predicted-Tokens-Per-S`, `X-Cale-Predicted-First-Token-S` | what Cale predicted for the size it chose |
| `X-Cale-Confidence` | `estimate` (Cale has seen no answer on this computer yet) or `learned` |
| `X-Cale-Goal-Met` | only when goals were given. The value is the string `true` or `false`. In a reply that is not streamed: whether the goals were met in the end, wait included. In a streamed reply the headers go out before the answer starts, so the value is only the prediction. The final value is `cale.goal_met_actual` in the last chunk. |
| `X-Cale-Unmet` | the goals that were not met (`speed`, `first_token`, `memory`, `quality`) |
| `cale.goals_actual` | an object keyed `speed`, `first_token`, `memory`, `quality` (only the goals you gave). Each holds the goal, the measured value (`actual`, or `actual_pct` for memory), and `met` (true, false, or null when it could not be measured). `speed` is answer tokens per second after the first token. `first_token` counts the wait in line. |
| `cale.goal_met_actual`, `cale.total_s` | the same in one value, and the total seconds (also in the last chunk of a stream) |
| `cale.size_changes` | sizes Cale changed to during the answer to keep a speed goal |
| `cale.answer_limit` | the limit on answer tokens that applied |
| `X-Cale-Memory-Goal-Met` | a model object with `memory_pct`: `true` or `false`, whether a size fit the memory goal when the object was made (see `memory_pct` in section 4) |
| `X-Cale-Total-S` | not streamed only: `cale.total_s` |

### The last chunk of a stream

A streamed reply is `data: {...}` events, then `data: [DONE]`. The text arrives in `choices[0].delta.content`. The last chunk before `[DONE]`
has an empty `delta`, the `finish_reason`, `usage` and a `cale` block. All the facts about the answer are there, because they are not known earlier.

| Field of `usage` in the last chunk | Meaning |
|---|---|
| `prompt_tokens`, `completion_tokens`, `total_tokens` | counts |
| `prompt_tokens_details.cached_tokens` | prompt tokens reused from the prompt cache, as in a reply that is not streamed (see "Prompt cache"). Streamed replies that use `tools` do not carry it yet. |

| Field of `cale` in the last chunk | Meaning |
|---|---|
| `size` | the size at the end (a percent) |
| `seconds` | seconds the engine worked on this answer (reading the prompt and writing the answer, not the wait in line) |
| `waited_s`, `total_s` | seconds in line, and wait plus `seconds` |
| `paused_pre_s`, `paused_dec_s` | seconds this answer was held back for an app of a higher tier, before its first word and after it (llama engine only; `null` or `0` elsewhere). They are part of `seconds`, but they are waiting, not work: Cale does not learn them as slow speed. |
| `goal_met`, `unmet`, `predicted`, `confidence`, `requested_size`, `memory_goal_met`, `answer_limit` | what Cale knew before the answer (the same as the headers) |
| `goal_met_actual`, `goals_actual` | the end result. `null` when no goal was given. |
| `size_changes` | the size changes during the answer, or `null` |
| `drafter`, `disk_mb`, `disk_s` | `drafter` is `false` today. `disk_mb` and `disk_s`: megabytes read from disk for this call, and the seconds spent on that. A stream with `tools` does not have them. |

A reply that is not streamed has the same `cale` fields, and also `first_token_s`, `switch_ms` and `answers_sharing_the_step`. A stream has no `first_token_s`.

### Who goes first when several apps ask at once

Cale decides this itself, with one rule that is the same on every computer. A developer does not manage it.

| Tier | Which calls | What happens |
|---|---|---|
| 1 (first) | The app asked for speed: `min_tokens_per_s` or `max_first_token_s` | Starts before waiting calls of the lower tiers. Is not held back. |
| 2 | No speed goal | Waits while a tier 1 answer runs. |
| 3 (last) | `power: "light"` and no speed goal (a background job) | Runs when no one in a higher tier needs the engine. |

Inside a tier the order of arrival counts. Time spent waiting shows in `waited_s` and `total_s` of the reply, so the app can see it.

| Engine | What a higher tier call gets |
|---|---|
| Windows and Linux (answers run together) | Lower tier answers that are running are paused at once and go on afterwards. The first word is not delayed by a background job. A library built before this rule cannot pause: it still orders the calls that wait. |
| Mac (one answer at a time) | A running answer is never stopped. The higher tier call starts next, ahead of every waiting call of a lower tier. It waits at most for the answer that is running now. |

### A client that leaves

If your app closes the connection while Cale is working (reading a long prompt, or writing the answer), Cale stops that work
at once. Other apps do not wait for an answer nobody will read. A request that is still waiting in line is dropped.

### Engine failure

If the engine fails during a call, you get `500 engine_error`. Cale unloads the model and loads it again on the next call.
`GET /cale/status` shows `last_error`.

## 4. Model objects and goals

A model object holds what your app needs. Cale returns an id. Use that id as `model` in chat calls. With `gemma-3n:auto`
you can send the same goals on one call, with a `cale_` prefix.

The id is in the reply (`id`): it is `name` in lower case, with other characters changed to `-`, cut to 30 characters. A second live object with the same name gets a suffix. Always read `id` from the reply. Use it as `model` in `POST /v1/chat/completions`. Only the app that made the object can use it. A call with a model object must not send `cale_*` fields (`400 goal_in_object`).

`POST /cale/models` with a JSON body. The reply stays open. The object lives while the connection is open.

| Field | Meaning |
|---|---|
| `name` | becomes the id |
| `size` | exact size: `"smart"`, `"fast"` or a percent. Not with goals. |
| `min_tokens_per_s` | answer at least this fast. Kept during the answer by changing the size between tokens. |
| `max_first_token_s` | first token within this many seconds of the call arriving (waiting in line and reading the prompt count) |
| `memory_pct` | keep Cale under this share of the computer's RAM, counting weights and context. It is a cap and a check: see "What `memory_pct` does" below. |
| `min_quality` | 0 to 1. No size that repeats less than this share of the full size's next-word choices. |
| `power` | `"normal"` or `"light"`. `"light"` limits the size to 75% or less, but only while the computer runs on battery or is hot. See "What `power` does" below. |
| `priority` | list of `"memory"`, `"quality"`, `"first_token"`, `"speed"`: which goal to keep when they conflict, most important first |
| `on_miss` | when no size meets the goals: `"best_effort"` (default), `"refuse"`, `"wait"` or `"try"`. `"smallest"` is the old name of `"best_effort"`. |
| `max_wait_s` | with `"wait"`: how long to keep trying, 0 to 60 s. Default 30. On an engine that answers one call at a time (a Mac), other apps wait too while Cale waits. On Windows and Linux, answers that already run go on, and this call is tried again a little later. |
| `allow_disk` | allow reading missing weights from disk. Default false. |
| `max_tokens` | the length limit of this object's answers. A `max_tokens` (or `max_completion_tokens`) in a chat call overrides it for that call. Cale uses the object's number only when the call sends none. |

#### What `memory_pct` does

It is both a cap and a check.

| Where | What happens |
|---|---|
| When the object is made | Cale keeps the largest size in RAM whose weights, plus the context of a hold (4096 tokens) and working room, fit in that share. This is a hold. Its size becomes the cap: no call with this object runs a larger size. If no size fits: with `on_miss: "refuse"` or `"wait"` you get `422 goal_missed` at once. With `"best_effort"` or `"try"` the smallest size is held and the memory goal counts as missed. |
| At every call | The memory a size needs for this prompt and answer (weights, context, working room) must be at most that share of the RAM. A size over it is a miss. `priority` says if the memory goal wins over the others (it is first by default). |
| With `cale_memory_pct` on `gemma-3n:auto` | Only the check at every call. There is no hold and no cap. |

What the reply says:

| Where | Value |
|---|---|
| `X-Cale-Memory-Goal-Met`, `cale.memory_goal_met` | a model object only: whether a size fit the share when the object was made |
| `cale.goals_actual.memory` | `{"goal_pct", "actual_pct", "met"}`. `actual_pct` is the memory the engine uses after the answer, as a share of the RAM (GPU memory on a Mac, process memory on Windows and Linux). `met` is true when it is at most `goal_pct`. |
| `cale.goal_met_actual`, `X-Cale-Unmet` | `memory` is listed when it was missed |

#### What `power` does

`"light"` has an effect only when the computer says it is on battery or hot. Then Cale uses no size above 75% (if all sizes are above it, the smallest). The test is made at each call.

| Computer | Counts as "light" |
|---|---|
| Mac | on battery, or the system has lowered the CPU speed for heat |
| Windows, Linux | on battery only (heat is not read). A desktop with no battery never counts. |

On a plugged-in cool computer, `"light"` changes nothing: the size is the same as with `"normal"`. A computer that cannot say its power state also counts as not light. `"light"` alone counts as a goal, so Cale chooses the size. `"normal"` is the default.

#### Cold start

The first call after an idle time includes loading the model (seconds). That time counts in `waited_s`, so it counts in `first_token` too. `max_first_token_s` is judged for each call, so a cold start can miss it (`goals_actual.first_token.met` is false, with the load time in `actual`). To avoid it, keep the model loaded: make a hold (section 5), or a model object with `size` or `memory_pct`, which also holds a size in RAM. A hold keeps the weights in RAM. Without any hold, the model unloads after 10 minutes with no call.

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

`POST /cale/load` with `{"size": ...}` or `{"memory_pct": n}` keeps a size in RAM. `POST /cale/unload` with `{"hold": id}` ends it early.
A hold keeps the model loaded, so the first call after a quiet time does not pay the load time (see "Cold start" in section 4). With `memory_pct`, `on_miss` is `"best_effort"` (default), `"refuse"`, `"wait"` or `"try"`; `"smallest"` is the old name of `"best_effort"`.

## 5b. Leases: model objects and holds over plain HTTP

Model objects and holds can live without an open connection, which suits any language and any HTTP library. Add `"lease_s": 5 to 3600`
to `POST /cale/models` or `POST /cale/load`. The reply comes at once. It has the usual object or hold fields and a `lease` field: `{"expires_in_s": N, "renew": ..., "end": ...}`. A `lease_s` outside 5 to 3600 gets `400 invalid_lease_s`. The thing lives for `lease_s` seconds after
its last renewal. If the app dies, it ends by itself. A chat call that uses a model object renews its lease too.

| Request | Does |
|---|---|
| `POST /cale/lease {"id": "<model id>"}` | renew a model object's lease |
| `POST /cale/lease {"hold": "<hold id>"}` | renew a hold's lease |
| `POST /cale/lease {"id": ..., "end": true}` or `DELETE /cale/models/<id>` | end a model object now |
| `POST /cale/lease {"hold": ..., "end": true}` | end a hold now |

Without `lease_s` the old form stays: the reply stays open and the object lives while the connection is open.

**A restart of Cale does not break your app.** Cale writes the spec of each live model object (its name and goals, never a prompt or an answer) to `objects.json` in its home folder. If Cale restarts (an update, a crash, a reboot) and your app then calls a model object that Cale no longer has, Cale brings it back from that file at the first call, as a leased object (600 seconds after its last use). Your app sees a normal answer. An object that ended, or whose lease ran out, is not brought back, and an object nobody used for 24 hours is forgotten. Only the app that made an object can use it.

## 5c. Clients

| Language | How |
|---|---|
| any | plain HTTP, with `curl` or any library: this document is the whole interface |
| Python | `cale/client.py`: `ensure_running`, `model`, `hold`, `predict`, `CaleError`. It has no `connect`: get the key with plain HTTP (section 2). |
| Node 18+ | `cale/client.mjs`: `connect`, `ensureRunning`, the `Cale` class, `CaleError`. See below. |

Python: copying `cale/client.py` alone into an app works. `ensure_running()` then runs the `cale` command (it also looks in Cale's install folder, and it uses `CALE_URL` as the address if set). The `hold(...)` helper has `on_miss="smallest"` as its default for `memory_pct`: `"smallest"` is the old name of `"best_effort"` and works.

Node (`import { connect, ensureRunning, Cale } from "./client.mjs"`):

| Call | Does |
|---|---|
| `await ensureRunning()` | Starts Cale if it does not answer. It runs the `cale` command (`cale start --wait`), so `cale` must be on the `PATH` of the app. If it is not installed, it throws. |
| `await connect("My App")` | Asks for a key and returns a `Cale` object. If the person has not answered, it prints the message (what the person must do) to `console.error` once, and polls every 2 s. `maxWaitS` (default 300) ends the wait with an error. Pass `onWait(msg)` to show the message somewhere else. It uses the `202` polling of section 2. |
| `new Cale(base, key)` | Reuses a key you saved: `new Cale("http://127.0.0.1:47821", savedKey)`. `connect` makes a new key and stops the old one (section 2), so save `cale.key` and use this next time. |
| `await cale.models()` | `GET /v1/models` |
| `await cale.predict({ ...goals, prompt_tokens })` | `POST /cale/predict` |
| `await cale.createModel({ name, leaseS, ...goals })` | A model object on a lease (section 5b). `leaseS` is 5 to 3600 s, default 60. Goals are the fields of section 4 (`min_tokens_per_s`, `memory_pct`, ...). Returns `{id, info, renew(), end()}`. A timer renews the lease every third of `leaseS`. |
| `await model.end()` | Ends the object now and stops the timer. Call it when the app is done. If the app dies, the lease ends by itself. |
| `await cale.hold({ size or memory_pct, leaseS })` | A hold on a lease. Returns `{id, info, end()}`. |
| `await cale.chat(body)` | Chat. Returns the JSON reply. |
| `for await (const chunk of cale.chatStream(body))` | Streamed chat. It sets `stream: true`. Each `chunk` is one parsed event. The last has `usage` and `cale`. |

A Node error is a `CaleError` with `status`, `code` and `body` (the error JSON).

## 6. List models

`GET /v1/models` (with key) lists sizes, your model objects, and what Cale predicts for this computer (`cale.this_computer`: speed and memory of every size, `floor`, `confidence`).

## Errors

All errors use `{"error": {"message", "type", "code", "param"}}`. Status codes: 400 bad request (including
`context_length_exceeded`), 401 bad key, 403 not allowed, 404 unknown model, 413 body too large, 422 goal missed
(with `on_miss: "refuse"` or `"wait"`), 429 busy, 500 `engine_error`, 507 `unsafe_for_this_computer`.
Some errors carry numbers in `error.cale`:

| Error | `error.cale` |
|---|---|
| `400 context_length_exceeded` | `prompt_tokens`, `max_tokens`, `max_context_tokens` |
| `422 goal_missed` in a chat call | `goal_met` (false), `unmet`, `predicted` (the best size: `tokens_per_s`, `first_token_s`, `memory_gb`, `quality`), `sizes`, `floor` |
| `422 goal_missed` when a model object is made with `on_miss: "refuse"` | the whole `/cale/predict` reply |
| `422 goal_missed` when a hold or an object with `memory_pct` finds no size | `goal` (`memory`), `goal_met`, `budget_gb`, `needs_gb` |
| `507 unsafe_for_this_computer` | `needed_gb`, `ram_gb`, `available_gb`, `smallest_size` |

A `422 constraint_dead_end` has no `cale` block.
