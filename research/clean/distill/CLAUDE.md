# distill/: the TREE track (read this first)

Goal: CALCULATE a smaller network (95 / 50 / 5% of neurons) from the big one, from weights and a generic input only. Networks: CLIP ViT-B/16 (main, closest to a language model) and ResNet-18. End goal: a language model Shonu has chosen, so every rule must be generic.

## Start of every session

1. Read `HANDOFF-TREE-WHOLE.md` (Shonu's words, the loop, Theory v1, the checks).
2. Read `THEORY.md` section 1. It is the concept of this track.
3. Read one essay passage in full: `../curves/map_agents/essay.md`.

## The concept binds. Old tracks do not

| Binds this track | History only (evidence, not rules) |
|---|---|
| `THEORY.md` section 1, `HANDOFF-TREE-WHOLE.md`, `NODE.md`, Shonu's words since 2026-11-25 | onecoarse / VECMATH (morph, "the coarse neuron is the parent", widened neuron), N1-N11, curves math mode, FANOUT rounds 1-3 knobs |

If an old note or an old statement conflicts with the concept, write the conflict in one line and ask Shonu. Never merge the two.

## How work runs

```
theory page (whole network, every part coarse) -> simplest build, both networks
   -> look whole vs the trained networks (+ leave-one-out rows inside the build)
   -> fix the code (same theory)  OR  change ONE line of the theory  -> repeat
```

| Do | Do not |
|---|---|
| Build every part coarse at once | Judge one coarse part inside a fine (pruned) network |
| Add a mechanism only as a line of the theory page | Stack fits, repairs or knobs to rescue a number |
| Start each loop with an essay passage | Run loops with no essay passage |
| Report the theory-page diff and the look table | Report a knob table |
| Keep going loop after loop | Stop to ask what is already decided here |

## Boundaries

- Calculation input: weights + generic input (`fanout/cache/input_pink_n256_s1.pt`, `fanout/cache/*_self_n256_k32_s1.pt`). Photos: scoring only.
- Trained students (`../neuron/results/coarse/r18/distill_whole/`, `clipkd/weights/`): answer key only, never a build input.
- Torch on the Mac only through `fanout/torchq.py`. GPU cluster jobs only with Shonu's yes.
- A claim needs: a prediction written first, an off row, a same-form null, the 1,000 test row and the 800-unseen read (`fanout/r3_critic_holdout.py`).

## Files

| File | Holds |
|---|---|
| `HANDOFF-TREE-WHOLE.md` | the current starting point |
| `THEORY.md` | the concept (section 1) and its record |
| `NODE.md` | node card and law readings (L1) |
| `ANALYSIS-2026-11-26.md` | why rounds drifted, and the habits to avoid |
| `FANOUT.md`, `fanout/` | the harness and the technical record of rounds 1-3 |
| `LOG.md` | trained targets (95 / 50 / 5%) |
