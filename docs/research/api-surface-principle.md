# API surface principle: power for the developer, simplicity by default

Raised by Shonu on 2026-10-15. Local AI lets a developer tune far more than cloud AI can. But every exposed knob makes a local app harder to write than a cloud app. Relative goals are the first-class primitive, because they let one app work on many machines.

## The rule

**A developer states what the app needs. Cale decides how.** Goals are public. Mechanisms are private.

| Kind | Examples | Where it lives |
|---|---|---|
| Goal (relative, machine-independent) | `min_tokens_per_s`, `max_first_token_s`, `memory_pct`, `min_quality`, `power`, `priority`, `on_miss` | Layer 1. The documented front door. |
| Capability (a thing only local can do) | keep a prompt warm, structured output, hold a model in memory, private data | Layer 1 or 2, always named by outcome ("keep this prompt warm"), never by mechanism ("KV cache type") |
| Mechanism | width, drafter, threads, GPU placement, KV cache type, in-answer step-down | Private. Reported, never required. |
| Pin (absolute, legacy) | `size`, fixed width | Layer 3. An escape hatch. |

## Three layers

| Layer | Who | What they write | Cost |
|---|---|---|---|
| 1. Goals | Every developer | 0 to 3 goal fields. Hello world works with none. | Minutes |
| 2. Policies | Developers with a special need | Named policies, for example `on_miss`, `on_stall`, `hard_floor` | One field each |
| 3. Pins and mechanisms | Researchers, legacy apps | Absolute size and similar pins | Developer accepts: the app may not be portable |

## Trust comes from seeing, not from steering

Give every call an explanation: which size Cale chose, why, what it predicted, what happened, and what would have met the goal. A developer who can see why trusts the auto mode. Control is the escape hatch. Visibility is the main path.

## Test for any new knob (all answers must be "yes" to go into Layer 1)

1. Can the developer state it without knowing the machine?
2. Does it describe an outcome the user feels (speed, wait, memory, quality, power)?
3. Does a sensible default exist, so the field is optional?
4. Can Cale check afterwards that it was met, and report it?

If any answer is "no", it is a mechanism. Keep it private, or put it in Layer 3.

## Where absolute needs fit

"Same intelligence everywhere" is itself a relative goal: `min_quality`. Use that before a size pin. A pin stays available, and the docs say it costs portability.

## What follows for work in progress

| Item | Public form |
|---|---|
| Prompt-copy drafter | None. Always on. Reported in the explanation. |
| In-answer step-down | One policy field with a floor, for example `on_stall`. Never a width number. |
| Passive sampling and the load model | None. Invisible. |
| Quantization | None. Cale ships one format per machine. |
