# Neuron checklist: open this before reading ANY neuron

Shonu, 2026-11-25: "a neuron is a verb, not a noun". This page collects the rules that were scattered over
`onecoarse/README.md` (reading rules 1-7), memory `feedback-think-interp.md`, and the essay
(`curves/map_agents/essay.md`). Do the steps in order. Do not write a job word before step 5.

## Step 0, before any test: the object (added 2026-11-25 after CALC 3.6)

Name the object and the rule that picks it, in a column "Object (picked by)" of every prediction table.

| Object | Picked by |
|---|---|
| a parent | named first, by its own rule (written before the run) |
| its children | the child rule R1: next-layer neurons that read the parent's write, in weights (`onecoarse/README.md` "Finding children") |
| co-firing units | shared top photos. Call them "co-firing units", NEVER "siblings", "children" or "branch" |

A tree test picks a parent first, then its children by R1. A fail on a substitute object is a fail of the object choice, never of the structure. CALC 3.6 broke this: it put tree names on co-firing units and blamed the tree.

## The steps

| # | Step | Essay line | Tool |
|---|---|---|---|
| 1 | Crops centred on the strongest position (conv: max cell, ViT: max token's 16 x 16 patch), shown beside the whole photo with the patch marked. Never whole photos alone | 253 "five strongest natural crops" | ViT: `distill/clipkd/crops.py`. Conv: `onecoarse/code/oc_mean.py` windows |
| 2 | Intersection image of the top crops (aligned colour mean and edge mean) BEFORE any words | 77, footnote 4 | `onecoarse/code/oc_mean.py`, ViT version in `distill/clipkd/crops.py` |
| 3 | Negative pole with the SAME tool (min pre-activation position). A separate reader reads it, without seeing the positive side. It is a gate, not an opposite. Numbers first: are the neuron's up-voting feeders on at its negative crops? | 233, 491 | same helper |
| 4 | Bands: top, strong, weak. The job is defined on top and strong only. Weak firings usually do not matter | 309, 353 | |
| 5 | Feeders: which upstream units drive the neuron on its top crops. All earlier MLP neurons and attention heads, not only the block before | 43 "look upstream", 231 votes | `distill/clipkd/read_poles.py` (pass B), `distill/clipkd/tree_verbs.py` (gate numbers) |
| 6 | Write two layers (README rule 3). (a) The VERB: pattern A against pattern B in shape, texture, layout and scale words, no object nouns. (b) The GROUPS it brings in from upstream, as noun examples | 101-105, footnote 3 | `onecoarse/code/verb_lint.py` on the Verb column |
| 7 | Every strong crop is explained: by the trunk, or by a branch traced back to the trunk. A crop I cannot explain is MY tool gap. Name it | 131-137, 205 | |
| 8 | A noun list that sounds unrelated is a STOP sign. Ask: "if I couldn't use groups of nouns at all, how would I describe this neuron?" Find the shared pattern before writing anything | 101-103 | |
| 9 | Class labels never group or select neurons, photos or firings. Never call a neuron "messy": my microscope failed | memory `feedback-think-interp` | `onecoarse/code/label_gate.py` |
| 10 | A reading is a hypothesis until a blind judge passes it (README rule 6). The judge gets both layers: the verb first, then the groups as examples. Held-out crops only, shuffled-words control, 24 targets per pole, two sheets, two judges | 391-401 | judge agent, code scores |

## Live tokens (ViT, added 2026-11-25 after CALC 3.6)

Before step 1, check which tokens are read downstream. With class-token pooling (open_clip `pool_type` = `tok`), a last-block firing on a patch token is read by nothing. In the last block, read the class token only. Compare two models only on live tokens of the same type.

## A reading table looks like this

| Neuron | Verb (pattern A vs pattern B, no nouns) | Positive groups (nouns allowed) | Negative pole (gate) | Top feeders | Unexplained strong crops |
|---|---|---|---|---|---|

## Worked wrong example (CALC 3.5, 2026-11-25)

| Student neuron | My job words | What was wrong |
|---|---|---|
| b8 kd721 | "hand-held devices: camera, gun, knife" | A noun list. "Camera + gun" reads as two unrelated things packed into one neuron. That is the superposition picture the essay rejects (lines 20-22, 101-105) |
| b12 kd699 | "maps, compass, dials, screens" | Same. No shared pattern asked for |
| b12 kd669 | "farm machinery and outdoor work" (corn + timber + tractors) | Same |

What I skipped: whole photos instead of max-centred crops (step 1), positive pole only (step 3), no intersection
image (step 2), no feeders (step 5), nouns in the job column (step 6). With the steps, the question is "what pattern
do camera and gun share at this neuron, and what does its negative side turn away?", not "which nouns does it hold?".
The V neuron is the model: cats and cars are one pointed shape, and its negative pole is an O (essay 73-77).

## Agents

The `map-reader` agent is tied to CrakBox tools and is marked "not usable" in its file. For a ViT or ResNet reading,
use a plain agent with Read and Write only, give it this page, the crop sheets and the intersection images, and ask
for the reading table above. One agent per pole.
