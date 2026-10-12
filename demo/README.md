# Demo: two apps, three computers, one runtime

Two ordinary open-source apps (Open WebUI and Aider) normally call a cloud AI. Here they call Cale instead.
Cale runs one local model on the computer. Each app says what it needs. Cale picks the size that fits this computer.
The same apps, with no change, run the full size on a fast computer and a smaller size on a weaker one.

| Computer | Engine | Starts Cale |
|---|---|---|
| Mac (Apple silicon) | MLX on the GPU | at login |
| Windows (Intel or AMD) | llama.cpp on the CPU | when an app needs it |
| Linux (Intel or AMD) | llama.cpp on the CPU | when an app needs it |

## What each app asks for

An app does not name a size. It makes a model object with goals. Cale does the rest.

| App | Model object | What it says | Used for |
|---|---|---|---|
| Open WebUI | `answers` | an answer of about 600 tokens in 90 s, using at most 40% of the RAM | chat answers |
| Open WebUI | `tasks` | an answer of about 60 tokens in 30 s | chat titles, tags, follow-up ideas |
| Aider | `code` | an answer of about 1200 tokens in 600 s (code must be right, so time matters less), using at most 40% of the RAM | the code |
| Aider | `commit` | an answer of about 60 tokens in 30 s | commit messages |

The numbers are the app developer's own. Change them with `CALE_ANSWER_DEADLINE`, `CALE_ANSWER_TOKENS`,
`CALE_TASK_DEADLINE`, `CALE_MEMORY_PCT`, `CALE_CODE_DEADLINE`, `CALE_CODE_TOKENS`, `CALE_COMMIT_DEADLINE`.

## Set up an app (same on every system)

Each app has its own Python environment. Make it once.

| System | Open WebUI | Aider |
|---|---|---|
| Mac, Linux | `uv venv apps/owui-venv && uv pip install --python apps/owui-venv/bin/python open-webui==0.11.4` | the same with `aider-chat==0.86.2` in `apps/aider-venv` |
| Windows | `python -m venv apps\owui-venv` then `apps\owui-venv\Scripts\pip install open-webui==0.11.4` | the same with `aider-chat==0.86.2` in `apps\aider-venv` |

## Run the demo

| Step | Type | What you see |
|---|---|---|
| 1 | Start Open WebUI: `apps/owui-venv/bin/python apps/open-webui/cale_webui.py` (Windows: `apps\owui-venv\Scripts\python apps\open-webui\cale_webui.py`) | A web page at `http://127.0.0.1:8810` with a card: "Get Cale" |
| 2 | Click **Get Cale** | A progress bar. The installer sets up Python, downloads the model, and starts Cale. |
| 3 | Click **Connect Open WebUI to Cale** | The computer asks "Allow Open WebUI to use Cale?". Click Allow. |
| 4 | Chat | The model is called `answers`. A small badge at the bottom left says which size Cale ran and whether the goal was met. |
| 5 | Start Aider in a git folder: `apps/aider-venv/bin/python apps/aider/cale_aider.py` | Aider finds Cale, asks once "Use it for Aider?", and needs **no download**. It shares the model. |

On the second app there is nothing to install. Both apps share the one loaded model.

## What to look for

| Computer | `answers` runs at | `tasks` runs at |
|---|---|---|
| Fast (Apple M5) | the full size, goal met | the full size, goal met |
| Weaker (Intel laptop, 4 cores) | a smaller size, chosen to meet the time | the smallest size, goal not met but still answered |

Check any time with `cale status`. `GET /v1/models` (with the app's key) shows each object and the size it ran at last.

## Test the demo on one computer

| Check | Type |
|---|---|
| Whole suite against a dev Cale | `python scripts/cale_dev_server.py /tmp/home`, then `python cale/tests/e2e.py --model`, `relative.py`, `objects.py` |
| Remove everything | `python cale/uninstall.py` |

## The HP test laptop (for the people who build Cale)

The Windows and Linux runs were done on one Intel laptop, reached from a Mac. See `scripts/hp` (runs a job on the laptop
and waits through link drops) and `scripts/hp_forward.sh` (shows a web page from the laptop on the Mac).
Run one system at a time on that laptop. It has 4 cores, so two Cales would slow each other and the speed numbers would be wrong.
Also, Windows and Linux (WSL2) share `localhost` ports there. A Cale or Open WebUI that is still running on one side answers for the other side. Stop one before you start the other.
