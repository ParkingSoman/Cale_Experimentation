# Cale principles

Read this before you change Cale. Reviewers read it too.

## The goal

Cale is one runtime and one strong model on a person's computer. Any app can use it.
The person downloads once. No pile of per-app models. No cloud subscription to try an AI tool.

We measure success with one equation:

> (number of app types that work **well**) × (number of consumer environments each one runs in)

| Factor | What it means |
|---|---|
| App types that work well | The gap between Cale and what a developer can build with a cloud AI API. Count what developers would really build, not only what is possible. |
| Environments | Operating systems × the compute each computer has: CPU only, GPU, small RAM, battery, other apps using the machine. |

Write once, run everywhere. An app written for Cale runs on every computer. Cale adapts. The app does not.

## Rules

| # | Rule | Why |
|---|---|---|
| 1 | An app states what it needs (goals). Cale picks the size per computer and per call. | The app writes once. |
| 2 | Cale pays the cost of per-OS and per-device work. App developers never see it. | More speed means more apps. |
| 3 | Use the fastest engine on each OS. Use generic code when it loses nothing. | Speed and reach both count. |
| 4 | Put shared logic above the engine interface (chat template, grammar, scheduler). | Two engines, one feature set. |
| 5 | Gemma 3n is the final model for this prototype. Its facts may be hard-coded, in one place. | Tuning for one model is the point. |
| 6 | Speed goals use tokens per second and first-token time. Never seconds per answer. | Answer length is unknown. |
| 7 | Cale never times the model at first launch. It predicts from hardware facts and live load, then corrects from real answers. | No wait for the user. |
| 8 | An unsupported request field returns a clear error. It is never ignored silently. | An app must know what it got. |
| 9 | Any language can use Cale over plain HTTP. | App types. |
| 10 | Nothing in `cale/` may exist only because a demo app needed it. | Overfitting lowers the equation. |
| 11 | No fix is "for one machine". Each fix is either general (Cale decides, same behavior everywhere) or a developer choice (the developer states it, Cale reports what ran). | A stronger future model puts strong computers in the same spot a weak computer is in today. |
| 12 | A developer must trust that their code behaves as expected on every computer. Cale never makes a behavior-changing trade-off per machine in silence. | Write once, run everywhere. |
| 13 | Never hard-code a size count, a layer count, or a device name or index. Logic works for any number of sizes, larger and smaller than today's. | Future Matryoshka models will have more sizes. |
| 14 | When no size can meet the goals, the behavior is defined and chosen by the developer (`on_miss`). One safety limit has no setting: Cale refuses any setup it predicts would crash or freeze the computer, with a documented error. | Weak computers must get a clear answer, not a crash. |

## How we check

- A reviewer agent reads only the public docs (`cale/README.md`, `docs/API.md`, this file).
- It writes probe apps that the builder never sees, and runs them on every computer we can reach.
- It returns a scorecard: app types that work well × environments.
- A second reviewer reads each change and asks: does this fit the demo apps or one machine?
