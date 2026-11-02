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
| 7 | Cale never times the model for an app. It measures this computer once, when Cale is installed (`cale tune`: a few minutes, about two on a fast Mac), and keeps the result. The person sees one line: "Getting Cale ready for this computer." After that it predicts from what it measured and the live load, and corrects from real answers. If nothing was measured (a test server, new hardware), the first load measures, once. `CALE_NO_MEASURE=1` turns that off. | Apps never wait for a measurement. A prediction starts from this computer, not from three test computers. |
| 8 | An unsupported request field returns a clear error. It is never ignored silently. | An app must know what it got. |
| 9 | Any language can use Cale over plain HTTP. | App types. |
| 10 | Nothing in `cale/` may exist only because a demo app needed it. | Overfitting lowers the equation. |
| 11 | No fix is "for one machine". Each fix is either general (Cale decides, same behavior everywhere) or a developer choice (the developer states it, Cale reports what ran). | A stronger future model puts strong computers in the same spot a weak computer is in today. |
| 12 | A developer must trust that their code behaves as expected on every computer. Cale never makes a behavior-changing trade-off per machine in silence. | Write once, run everywhere. |
| 13 | Never hard-code a size count, a layer count, or a device name or index. Logic works for any number of sizes, larger and smaller than today's. | Future Matryoshka models will have more sizes. |
| 14 | When no size can meet the goals, the behavior is defined and chosen by the developer (`on_miss`). One safety limit has no setting: Cale refuses any setup it predicts would crash or freeze the computer, with a documented error. | Weak computers must get a clear answer, not a crash. |
| 15 | No number that depends on the computer is written in the code. Ask the computer (the OS, the device, the engine) or measure it, and keep what was measured. A number that is a policy (a limit the developer may want to change) is named as a policy, has one place, and is reported. `docs/CONSTANTS.md` lists every number and which kind it is. | Many environments: a constant tuned on two computers is a guess on the third. |
| 16 | The equation applies to every moment around Cale, not only to apps: the download, the install, each open of Cale, each open of an app, each message, each change of intelligence level. Each moment must feel the same on a weak computer and on a strong one. Only the range of intelligence levels shifts per computer. | Same feel everywhere is the point of relative goals. "The computer is old, so it is slow" is never accepted. The only accepted limit is the small end of the size range (research). |
| 17 | The window never waits for the AI. Opening Cale or an app must not depend on the model loading. A message sent while the AI loads waits, with a visible sign, and is answered. | The model load is the part that varies most between computers. |
| 18 | By default, text flows at reading speed or better, and no faster than needed. Speed past reading pace is spent on a larger size. The Quicker/Smarter slider lets a person go beyond that: its smarter end is the largest size that fits, even if text is a little slower than reading speed. The default position never is. | Overshoot is an error (more intelligence was available). The default is the same feel everywhere. The ends are the person's choice. |
| 19 | Work that does not depend on the computer is done once, before the person installs, and shipped ready. This holds for model preparation (the repacked model file is hosted: `parkingsoman/cale-gemma-3n-E4B-it-GGUF`) and for any other preparation. Only work that needs this computer (the hardware test, the choice of engine) runs at install. | A person's computer must not repeat work that is the same everywhere. It costs the most on the weakest computer. |

## How we check

- A reviewer agent reads only the public docs (`cale/README.md`, `docs/API.md`, this file).
- It writes probe apps that the builder never sees, and runs them on every computer we can reach.
- It returns a scorecard: app types that work well × environments.
- A second reviewer reads each change and asks: does this fit the demo apps or one machine?
