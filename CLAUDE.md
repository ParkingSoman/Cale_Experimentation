# Cale: read this first

## What Cale is

Cale is ONE runtime that a consumer downloads ONCE. After that it is invisible.
Any app built for Cale then works on that computer. The consumer does not pick models, run commands, or see numbers.
Developers state goals ("first word in about a second"). Cale picks the model size that meets them on each computer.
The aim is ecosystem buy-in: many developers build for Cale, many consumers already have it.
Success = (app types that work well) x (consumer environments each one runs in). Judge every design by it.

## Rules for suggestions

| Rule | Meaning |
|---|---|
| Never bundle Cale inside an app | Cale is a shared install. An app is a small thing that uses it. |
| The consumer never sees anything technical | No tokens/s, model sizes, "fitting", Terminal commands, or the word "model" in consumer screens. If a demo screen gets technical, the demo failed. |
| The Cale Hub is a developer view | Do not show it to the consumer or in the demo. |
| Gemma 3n is a prototype | The planned model is a Matryoshka model built for laptops. |

## Demo decisions already made (do not re-ask)

| Decision | Detail |
|---|---|
| Demo apps | Quick Notes plus ONE new second app (a different app type, also on relative goals, works on both computers). The other four old apps are for the funding pitch: the size range is too narrow for them. |
| Second-app purpose | Show that one Cale download lets people start other Cale apps with no hassle, and the download happens once. |
| Show only the relative approach | Do NOT show absolute against relative live. Say it, do not stage it. |
| Core message | Same app code, different operating systems, Cale picks the model for the app's goals. The app feels the same on a recent Mac and a ten-year-old laptop. |
| No commands in the demo | The presenter types nothing. Anything that needs a command is a gap to fix. |
| Browser checks | Use Claude in Chrome only. |
| Cale is a real program (decided 2026-10-17) | Cale has its own windows and its own chat, like ChatGPT or Claude. No localhost page, no address bar, no console window. Cartoony, down-to-earth look (not sci-fi): the mascot is a carrot in sunglasses, in orange and cream. The green kale look is retired. Tokens and rules: `cale/brand/README.md`. It applies to Cale's own screens and chat only. |
| Cale's chat base | AnythingLLM (MIT licence), rebranded as Cale. Not Open WebUI: its licence allows removing its branding only for 50 or fewer users. Open WebUI stays in the funding pitch as an unchanged third-party app. |
| Demo story | Install Cale once, chat at once (no sign-up, no usage limits). Quick Notes is a downloaded desktop app. Rewrite is a real website. All three use the same AI on this computer. |
| Start state | The Cale installer and the Quick Notes installer are already downloaded. The install itself is a recorded time-lapse. |

Run sheet and measured numbers: `demo/README.md`. Latest handoff: `docs/HANDOFF-2026-10-15.md`.
