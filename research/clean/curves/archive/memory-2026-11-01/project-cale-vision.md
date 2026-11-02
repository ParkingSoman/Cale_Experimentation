---
name: project-cale-vision
description: "What Cale is for (2026-10-12): the success equation, one runtime + one strong Matryoshka model, why overfitting and CPU-only fail; Gemma 3n is the prototype final model"
metadata:
  type: project
---

Shonu's equation (the "Cale equation"): **success = (number of app types that work WELL) x (number of consumer environments each one runs in).** Write once, run everywhere.

- App types: judged by the gap between Cale and what developers can do with cloud AI. Count what developers would REALLY build, not only what is possible (nobody puts sensitive work on a weak model, so model quality limits app types).
- Environments: operating systems x compute available (CPU only, GPU, small RAM, battery, other apps loading the machine).

**Why:** a user downloads ONE runtime and ONE strong model once. Any app uses it. No pile of per-app local models. No cloud subscription just to try an AI tool. Later Shonu hopes to train his own Matryoshka model built for laptops.

**How to apply:** use the equation to judge every design choice and to question earlier decisions (he invited this). Overfitting to the demo apps (Open WebUI, Aider) lowers the first factor. CPU-only on Windows/Linux, no arm64, a Python-only client lower the second. Shonu will test me by building many diverse apps on Cale only. Plan: `~/.claude/plans/you-were-finishing-up-splendid-willow.md` (Phase 6, 8 stages).

## Gemma 3n is the final model, for the prototype (2026-10-12)
- Pretend Gemma 3n is the final model; when he has a better one he optimizes Cale for that one. Optimizing for exactly one model lets Cale do things Ollama and LM Studio cannot (exact bytes per token, exact KV size, which layers each size skips, nested drafter, size switch between tokens, shared prompt cache).
- Hard-coded Gemma 3n facts (35 layers, `e4b`/`e2b`) are NOT overfits. Do not build a model manifest or swappable-model layer. Facts live in one place (`cale/gemma3n_facts.py`). A future model gets its own tuning.

See [[feedback-cale-design-rules]], [[feedback-testing-rules]], [[project-cale-phases-3-5]], [[project-research-history]].

## For the research track (2026-10-31)
Shonu: "the cale vision is really just about instant computation given a defined zoom. we will expand to language
models later but just pretend for now resnet is the thing we need". Research end state = any level of a defined
ResNet zoom computed instantly, optimal within a proved bound. No app or runtime work in that roadmap
(research/clean/curves/ROADMAP.md).
