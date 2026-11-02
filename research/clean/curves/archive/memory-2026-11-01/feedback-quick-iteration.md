---
name: feedback-quick-iteration
description: Shonu wants quick iteration loops (1-3 min each); build a small cache once, never launch hour-long sweeps while iterating on a method
metadata:
  type: feedback
---

When iterating on a method, keep every loop to about 1-3 minutes. Build a small cached test set once (minutes, not an hour), then iterate on it. Do not launch long multi-family / multi-arm sweeps (30-80 min) mid-iteration. Scale up only after the method has settled.

**Why:** Shonu said it repeatedly (2026-10-21: "these runs are taking too long. we need fast iteration"; 2026-10-22: "you shouldn't be doing hour long runs ... this is a common error you make"). Long runs also pile up and fight for the 24 GB of memory.

**How to apply:** before launching, estimate the run time. If above ~5 min while iterating, shrink it: fewer families, fewer images, cached tensors on disk, one stage. Use `python -u` (or flush) so background logs show progress, and never pipe a background job through a buffering grep. Related: [[feedback-research-method]], [[lesson-rules]].

- 2026-10-23 night (repeat): "you're allowed to do long runs of course but ... most of the time when you do them a
  shorter experiment where you had quicker iteration on the math would've served you better." Trigger: a 31-min
  InceptionV1 generation (1024 boundary inputs) before the idea was tested on the toy. Rule: test each new idea on the
  toy (`research/clean/curves/toy/`, seconds per loop) or on a 64-input InceptionV1 slice FIRST; a long run only
  confirms a settled idea. While a long run is going, iterate the math on the toy, not wait.

## Smoke run first, one GPU job, math before solver variants (Shonu, 2026-10-24 night)
"we'd likely benefit from shorter runs which precede these longer runs so we don't wait a bunch of time before just
realizing the run itself is fundamentally flawed" + "smaller test sizes, math which needs no gpu, and one job at a time".
**How to apply:** every run > 3 min gets a ~1-min smoke run first (tiny n, 1-2 iterations, check shapes, signs, first
numbers). ONE GPU job at a time. Prefer CPU analysis of cached tensors. After 2 solver/variant runs in a row, stop and
write the paper statement (the 2026-10-24 night C4-C6 chain was engineering drift; also 3 runs died on dtype/arg bugs a
smoke run would have caught). Generators must checkpoint per batch.

**Heartbeats on every long run (2026-11-02).** Shonu: "you should have heartbeats for these runs so we know it is
running well. do something like 15 mins". Rule: any background run expected to take more than ~10 minutes gets a
Monitor at launch that prints, every 15 minutes, alive/CPU/last result line, and prints at once on a new result, on
finish, and on crash (grep Traceback/Error). Monitors expire at 30 min: re-arm while the run lives.
