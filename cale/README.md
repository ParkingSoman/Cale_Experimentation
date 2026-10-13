# Cale: a local AI runtime for Mac, Windows and Linux

Install it once. Apps then call the AI on this computer with a local key, the way they call a cloud API.
Cale holds one model. That model contains a family of sizes (Gemma 3n: the small size lives inside the large one).
Your app says what it needs. Cale picks the size that fits this computer.

## For the user

| Type | What happens |
|---|---|
| `./cale/install.sh` (Mac, Linux) | Installs Cale, downloads the model once, sets up the `cale` command |
| `powershell -ExecutionPolicy Bypass -File cale\install.ps1` (Windows) | Same |
| `python cale/install.py` | Same, on any system that has Python 3.9 or newer |
| `cale status` | Shows if Cale runs and how much memory it uses |
| `cale apps` | Lists apps that hold a key |
| `cale revoke NAME` | Removes an app's access |
| `cale start`, `cale stop`, `cale logs` | Control Cale |
| `./cale/uninstall.sh`, `cale\uninstall.ps1`, `python cale/uninstall.py` | Remove everything |

| | Mac | Windows | Linux |
|---|---|---|---|
| Needs | Apple silicon, macOS 14 or newer | 64-bit Intel or AMD (older ones without AVX2 run slower), or ARM64 | 64-bit Intel or AMD (older ones without AVX2 run slower), or ARM64 |
| Engine | MLX on the GPU | llama.cpp on the CPU | llama.cpp on the CPU |
| Model download | about 5.5 GB | about 4.4 GB | about 4.4 GB |
| Starts | at login (LaunchAgent) | when an app needs it | when an app needs it |
| Allow question | a dialog | a dialog | a dialog (needs `zenity` or `kdialog` and a screen) |
| Install folder | `~/Library/Application Support/Cale` | `%LOCALAPPDATA%\Cale` | `~/.local/share/cale` |

On Windows and Linux there is no background service. An app starts Cale with `cale.client.ensure_running()`.
Cale ends itself after 10 minutes with no app using it. The install adds no scheduled task and no login item.

The first time an app asks to use Cale, the computer shows an Allow question.
The model loads when an app first needs it and unloads after 10 minutes idle.
If memory runs low, Cale first caps the size it runs. Only then does it unload what no app holds.

Install settings (environment variables, all optional):

| Setting | Meaning |
|---|---|
| `CALE_HOME` | install folder |
| `CALE_MODEL_SRC` | a model you already have (a folder on Mac, a `.gguf` file elsewhere). It is linked, not copied. |
| `CALE_MAX_MBPS` | limit the download speed, in megabytes per second. Use it on a shared Wi-Fi. |
| `CALE_ALLOW_COPY=1` | copy the model if a link is not possible |
| `CALE_PROGRESS=1` | print `CALE_PROGRESS {...}` lines that an app can show |

## For the developer: any language (curl, Node, Go, Rust, a shell script)

Cale is plain HTTP. `docs/API.md` is the whole interface. A complete session with `curl`:

```sh
cale start --wait                                            # starts Cale if it is not running
curl -s -X POST http://127.0.0.1:47821/cale/connect -d '{"app":"My Script","wait_s":0}'
#   -> {"status":"pending","ticket":"t-...","poll":"/cale/connect/status?ticket=t-...","message":"..."}
#   the person allows it (a dialog, or `cale allow "My Script"` in a terminal if the computer has no screen)
curl -s http://127.0.0.1:47821/cale/connect/status?ticket=t-...   # -> {"api_key": "cale-..."}  (once)
curl -s -X POST http://127.0.0.1:47821/v1/chat/completions -H "Authorization: Bearer cale-..." \
  -d '{"model":"gemma-3n:auto","cale_min_tokens_per_s":15,"messages":[{"role":"user","content":"Hello"}]}'
```

| Command | Does |
|---|---|
| `cale start --wait` | starts Cale and returns when it answers |
| `cale status --json` | status for programs |
| `cale computer` | what Cale predicts for this computer: the speed and memory of every size |
| `cale pending`, `cale allow NAME`, `cale deny NAME` | approve apps on a computer with no screen |

Node apps copy `cale/client.mjs` (no packages). A model object on a lease lives without an open connection: pass `lease_s` (see `docs/API.md`, section 5b).

## For the developer: model objects and goals

A model object is the model you want, written as goals. Cale returns a name. You use that name as the
`model` in normal OpenAI calls. Cale picks the size for every call, on every computer.

```python
from cale.client import model, predict
from openai import OpenAI

with model(base_url, cale_key, name="answers", min_tokens_per_s=15, max_first_token_s=3, memory_pct=40) as m:
    print(m.info["prediction"])                     # can this computer meet the goals? which size? what does every size do?
    client = OpenAI(base_url=base_url, api_key=cale_key)
    client.chat.completions.create(model=m.id, messages=[...])      # m.id is "answers"
```

### The goals

| Goal | Meaning |
|---|---|
| `min_tokens_per_s` | Answer at least this fast (tokens per second). Kept during the answer: if the live speed falls under it, Cale steps to a smaller size between two tokens, and back up when there is room. |
| `max_first_token_s` | The first token within this many seconds after the call arrives. Time waiting in line and reading the prompt count. |
| `memory_pct` | Stay under this share of the computer's RAM. The weights and the context both count. |
| `min_quality` | Use no size that repeats less than this share (0 to 1) of the full size's next-word choices. `0.95` means "almost the full model". |
| `power` | `"normal"` (default) or `"light"`: on battery or when the computer is hot, run a smaller size. |

Other settings of an object:

| Parameter | Meaning |
|---|---|
| `name` | Becomes the id (`answers`). A second live object with the same name gets a suffix. |
| `size` | An exact size: `"smart"`, `"fast"` or a percent. Do not combine it with goals. |
| `max_tokens` | The length limit of your answers. If you set none, only the context limit applies. |
| `priority` | Which goals to keep when they cannot all be met, most important first. Default: `["memory", "quality", "first_token", "speed"]`. |
| `on_miss` | What to do when no size meets the goals: `"best_effort"` (default, runs the size that misses least and says what missed), `"refuse"` (422, nothing runs), `"wait"` (try again for `max_wait_s`, 0 to 60, default 30, then 422), `"try"` (run anyway: also allows disk reads). |
| `allow_disk` | Let a call read missing weights from disk. Default `False`. |

`deadline_s` was removed. Cale cannot know how long an answer will be, so it cannot keep a deadline for a whole answer.

### Ask before you run

`predict(base_url, key, min_tokens_per_s=15, prompt_tokens=500)` (or `POST /cale/predict`) runs nothing. It says:

| Field | Meaning |
|---|---|
| `goal_met`, `unmet` | Whether this computer can meet the goals, and which ones it cannot. |
| `chosen` | The size Cale would use, with the predicted `tokens_per_s`, `first_token_s`, `memory_gb`, `quality`. |
| `sizes` | The same numbers for every size. |
| `floor` | The fastest size and the smallest size on this computer: the most this computer can do. |
| `confidence` | `estimate` (no answer seen yet) or `learned` (corrected by real answers). |
| `unsafe` | `true` if no size can run this safely. |

Cale does not time the model on first launch. It predicts from the speed of this computer's memory (a 0.15 s probe), from
facts about Gemma 3n, and from what the computer is doing now. Every real answer corrects the numbers, and Cale keeps them
for the next start.

### The safety limit

Cale never runs what it predicts would freeze or crash the computer. This holds for every setting, `on_miss="try"` too,
and for exact sizes. The reply is `507 unsafe_for_this_computer` with the memory needed and what is free. Handle it like any
other error: use a shorter prompt, or ask the person to free memory.

### What a reply tells you

| Where | Field |
|---|---|
| Headers | `X-Cale-Size`, `X-Cale-Goal-Met`, `X-Cale-Unmet`, `X-Cale-Predicted-Tokens-Per-S`, `X-Cale-Predicted-First-Token-S`, `X-Cale-Confidence`, `X-Cale-Disk-MB`, `X-Cale-Waited-S` |
| Body `cale.goals_actual` | for each goal: the goal, what really happened, and whether it was met |
| Body `cale.size_changes` | the size changes made during the answer, if any |
| Body `cale.answer_limit` | the answer length limit that applied |

How it behaves:

| Rule | Detail |
|---|---|
| Life | The object lives while the `with` block runs, or until the app quits. One open connection holds it. |
| Memory | An object with `size` or `memory_pct` also keeps that size in RAM while it lives. |
| Per call | Cale picks the largest size that meets the goals. The choice is made for every call. |
| Weak computer | The same object runs a smaller size. Your app does not change. |
| Visibility | `GET /v1/models` lists your objects, with the size each ran at last (`cale.last`). |
| Privacy | An app sees only its own objects. |
| Request fields | Do not send `cale_*` fields with an object. The object holds its own goals. |

An app can start Cale itself before it makes objects:

| Call | What it does |
|---|---|
| `from cale.client import ensure_running` | Starts Cale if it is not running. Raises `CaleError` if it does not come up. |

## Absolute sizes (for completeness)

| `model` | Size used |
|---|---|
| `gemma-3n:smart` (or `gemma-3n`) | the full model, 100% |
| `gemma-3n:fast` | the smallest trained size, 48% |
| `gemma-3n:<48-100>` | the trained size nearest to that percent |
| `gemma-3n:auto` | Cale chooses. Add the goals with a `cale_` prefix in `extra_body`: `cale_min_tokens_per_s`, `cale_max_first_token_s`, `cale_memory_pct`, `cale_min_quality`, `cale_power`, `cale_priority`, `cale_on_miss`, `cale_max_wait_s`, `cale_allow_disk`. |

An absolute size is the same on every computer. A goal adapts. Use a goal or a model object
when your app runs on computers you do not know.

`temperature` defaults to 0. Text, and pictures on a Mac (see `docs/API.md`, Pictures). Cale supports `tools`, `tool_choice` and `response_format` (JSON mode and JSON schema, always valid): see `docs/API.md`. Call Cale from your app's backend: it refuses requests that carry a browser `Origin` header.

## Holds

A hold keeps a size in memory. Smaller sizes live inside the larger ones, so a held large size makes the small sizes free.

| Call | What it does |
|---|---|
| `with hold(base_url, key, size="smart")` | Keeps that size in memory while the block runs |
| `with hold(base_url, key, memory_pct=25)` | Keeps the largest size that fits in 25% of the RAM |
| `POST /cale/load` | The raw call. The reply stays open. |
| `POST /cale/unload {"hold": "<id>"}` | Ends a hold early |

A hold ends when you close the connection or the app quits. Memory pressure does not end a hold.
An app with no hold still works. Its first call keeps the full size in memory until 10 minutes after its last call.

If a call needs a size larger than what is held, Cale reads the missing weights from disk for that call.
The header `X-Cale-Disk-MB` says how much. The memory is freed when the call ends. A call never fails for this reason.

## What Cale knows about each computer

Cale predicts speed and learns from real answers. It never times the model on purpose.
`GET /v1/models` shows this computer (`cale.this_computer` in the JSON; `this_mac` is the old name for it): the predicted speed of
every size, the memory each needs, its quality, the `floor`, and how sure Cale is.

## Where things live

| What | Where |
|---|---|
| Install folder | see the table above (`CALE_HOME` overrides it) |
| Runtime, model, library, logs | `runtime/`, `models/`, `lib/`, `logs/cale.log` in the install folder |
| Keys | `keys.json` (hashes only) |
| What Cale learned about this computer's speed | `predict.json` |
| Address | `http://127.0.0.1:47821` (`CALE_PORT` changes it) |

## For people who build Cale

| Part | File |
|---|---|
| All OS-specific code | `cale/oslayer.py` |
| Start and stop without a service manager | `cale/service.py` |
| The engines | `cale/backends/` (`llama.py`, and `nest/gemma3n.py` for MLX) |
| The patched llama.cpp, with sizes | `llama/` (see `llama/README.md`) |
| Tests | `cale/tests/` (see below) |

| Test | Needs | Run |
|---|---|---|
| `oslayer.py` | nothing | `python cale/tests/oslayer.py` |
| `choose.py` | nothing | `python -m cale.tests.choose` |
| `allow_dialog.py` | a screen and a person | `python cale/tests/allow_dialog.py`: shows the real Allow question and says what you clicked |
| `backend_llama.py` | the library and a `.gguf` | `python cale/tests/backend_llama.py MODEL.gguf LIB` |
| `e2e.py`, `relative.py`, `objects.py` | a running dev server | `python scripts/cale_dev_server.py HOME`, then `python cale/tests/<name>.py` |
