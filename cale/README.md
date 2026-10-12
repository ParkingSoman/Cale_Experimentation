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
| Needs | Apple silicon, macOS 14 or newer | 64-bit Intel or AMD with AVX2 | 64-bit Intel or AMD with AVX2 |
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

## For the developer: model objects

A model object is the model you want, written as parameters. Cale returns a name. You use that name as the
`model` in normal OpenAI calls.

```python
from cale.client import model
from openai import OpenAI

with model(base_url, cale_key, name="answers", deadline_s=90, max_tokens=600, memory_pct=40) as m:
    client = OpenAI(base_url=base_url, api_key=cale_key)
    client.chat.completions.create(model=m.id, messages=[...])      # m.id is "answers"
```

| Parameter | Meaning |
|---|---|
| `name` | Becomes the id (`answers`). A second live object with the same name gets a suffix. |
| `size` | An exact size: `"smart"`, `"fast"` or a percent. Do not combine it with goals. |
| `deadline_s` | Finish one answer in this many seconds. Needs `max_tokens`. |
| `max_tokens` | How long your answers are. The deadline is judged against it. It is also the default limit. |
| `memory_pct` | Keep Cale's memory use under this share of the computer's RAM. |
| `allow_disk` | Let a call read missing weights from disk. Default `False`. |
| `on_miss` | `"smallest"` (default): run the smallest size and report. `"refuse"`: return 422 with an estimate. |

How it behaves:

| Rule | Detail |
|---|---|
| Life | The object lives while the `with` block runs, or until the app quits. One open connection holds it. |
| Memory | An object with `size` or `memory_pct` also keeps that size in RAM while it lives. |
| Per call | With a deadline, Cale picks the largest size predicted to finish in time. The choice is made for every call. |
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
| `gemma-3n:auto` | Cale chooses. Add `cale_deadline_s`, `cale_allow_disk`, `cale_on_miss` in `extra_body`. |

An absolute size is the same on every computer. A goal adapts. Use a goal or a model object
when your app runs on computers you do not know.

`temperature` defaults to 0. Text only for now. Call Cale from your app's backend: it refuses requests that carry a browser `Origin` header.

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

## What Cale measures on each computer

Cale measures a profile of this computer once, when a goal first needs it. It times three sizes and fits the rest.
It saves the profile and corrects it from real answers. The time prediction has a 25% safety margin.
Answer headers: `X-Cale-Size`, `X-Cale-Predicted-S`, `X-Cale-Goal-Met`, `X-Cale-Disk-MB`.

`GET /v1/models` shows the profile of this computer (`cale.this_computer` in the JSON; `this_mac` is the old name for it): speed and memory of every size.

## Where things live

| What | Where |
|---|---|
| Install folder | see the table above (`CALE_HOME` overrides it) |
| Runtime, model, library, logs | `runtime/`, `models/`, `lib/`, `logs/cale.log` in the install folder |
| Keys | `keys.json` (hashes only) |
| Profile | `profile.json` |
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
