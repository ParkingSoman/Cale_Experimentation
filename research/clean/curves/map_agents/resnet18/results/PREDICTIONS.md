# Predictions, written before the first full data pass (2026-11-03)

These are mine. Each one can fail. Each gets a line here when its number arrives.

## One channel, one job across a stage (Step 1)

| Stage | Skip share of the variance at block 1 | Correlation of a channel, block 0 against block 1 |
|---|---|---|
| s4 | about .88 (card 3-34 measured this for the last block) | median above .7 |
| s3 | .6 to .85 | median above .6 |
| s2 | .5 to .8 | median above .6 |
| s1 | .5 to .8 | median above .6 |

If a stage has a median correlation under .5, the unit becomes the block in that stage.

## The test slice (Step 4), before any agent runs

| Line | Pass line | My prediction | Why |
|---|---|---|---|
| Specific enough to lose | 85% of channels show a gap of at least .2 | Passes at s4 and s3. Near the line at s2 and s1 | The pilot's mean gap was .29 at s4. Texture channels were its weakest |
| Real parents, 1 of 5 | at least 70% | 50 to 75%. It may fail | Broad parents are hard to tell from distractors by card alone |
| Parents act, against matched controls | 2 times more | Passes on most links | The feeder table selects strong edges, so this line favors the agent. Removal effects could still be spread thin |
| Families act | Predicted classes among the top losers | Passes for tight families. Weak for broad ones | |

## The use test (Step 6), before any map exists

| Line | Pass line | My prediction |
|---|---|---|
| The map earns its place | KL at least 1.5 times below both controls, at three sizes, on dogs and cats | Beats random groups clearly. Against strongest channels, it is close and may fail, since the old tree on ConvNeXt tied with energy as a removal order |

# Results

## One channel, one job across a stage (2026-11-03; `job_table.json`, `skip_check.log`)

| Stage | Predicted skip share at block 1 | Measured, variance (median) | Measured, skip / z at the peak | Predicted correlation | Measured (median) | Top 100 photos shared |
|---|---|---|---|---|---|---|
| s4 | about .88 | .01 | .07 | above .7 | .55 | 12 |
| s3 | .6 to .85 | .42 | .77 | above .6 | .82 | 39 |
| s2 | .5 to .8 | .51 | .70 | above .6 | .88 | 33 |
| s1 | .5 to .8 | .43 | .71 | above .6 | .75 | 32 |

**My s4 prediction failed badly.** It rested on card 3-34 and LADDER 45. Those read the feeder table, where the entry for a channel's own number looks large. That entry is the skip plus block 1's residual path reading the same channel. At s4 the skip itself is .07 of z at the peak.

The second prediction, in `skip_check.py`, held at s4: skip under .2 of z, and own share of input x gradient above that (.21 against .07). At stages 1 to 3 it predicted .3 to .7, and the measured values are .70 to .77, at or just above the top.

**Decision (technical, by delegation):** the stream channel stays the unit, read at block 1. No stage has a median correlation under .5, which was the line fixed in advance. Stage 4 is close (.55), and its top photos differ, so links inside stage 4 count as real links.

## The test slice, first scoring (2026-11-03; sealed photos; `blind_a.json`, `blind_b.json`, `acts_sealed.json`, `families.json`)

106 channels, 119 links, 17 families, 15 of them with stage 3 or 4 channels and so in the family test. About 0.8M tokens for the reading.

| Line | Pass line | Prediction | Result | Verdict |
|---|---|---|---|---|
| Specific enough to lose | 85% at gap >= .2 | Passes at s4 and s3. Near the line at s2 and s1 | 44% overall. s4 80%, s3 29%, s2 20%, s1 13%. Mean gap s4 .34, s3 .13, s2 .10, s1 .08 | FAILS. Prediction wrong at s3, s2, s1 |
| Real parents, 1 of 5 | 70% | 50 to 75% | 57%. Same agent wrote both cards 57%, different agents 54%, so wording leak looks small | FAILS. Prediction held |
| Parents act, 2 times the controls | no share fixed; "most links" predicted | 70 to 90% | 92% of 119 links. s1 100%, s2 75%, s3 100%, s4 90% | PASSES, above the prediction |
| Effect of one parent at s4 | | 2 to 20% of z | Median 3.8% | Held, at the low end |
| Error keeping k feeders | | k = 4 above .5, k = 16 about .2 to .5, k = 64 under .2 | Share of z^2 changed at s4: .97, .91, .77 at k = 4, 16, 64. s3: .72, .55, .21. s2: .56, .20, 0 (the median, with 64 of 128 kept). s1: .08, .04, 0 (64 is the whole layer) | FAILS at s4 and s3 |
| Families act | most of 15 families: 6 of 10 predicted classes in the top 100 losers, and 2 times the controls on them | About half pass. Tone and strand pass. Furry face does not | 5 of 15 pass. Mean 4.9 of 10 (chance 1). KL on the predicted classes beats the controls by 2 times or more in 14 of 15. Tone and strand passed, furry face failed | FAILS the line. The family predictions held |

Change of measure, said plainly: the script's fixed measure divides by the variance of z over the child's top 64 photos. That variance is tiny, so the numbers run into the hundreds and say little. They are in the JSON. I read the share of z^2 in its place, which the same script also recorded before any result was seen.

Bug found and fixed before these numbers: the tools filled a channel at both blocks with its block 1 mean. Definition DC1 needs each layer's own mean. `acts_sealed_wrongmean.json` holds the old run: 93% against 92%, so the effect is small. The agents' own `acts` numbers on their cards came from the old fill.

## The cut check, TC1's premise on real cuts (`cut_check.json`; dogs and cats goal, sealed photos)

| Cut | Kept at s3, s4 | Median eps_par at s3 b1, s4 b0, s4 b1 | KL | Agreement | Random (3 seeds) | Strongest |
|---|---|---|---|---|---|---|
| dogs: keep only the dog and cat cards | 22, 20 | .71, 1.24, .91 | 5.88 | .001 | 5.71, 5.79, 5.80 | 6.29 |
| novehicle: remove the vehicle cards | 243, 492 | .017, .037, .033 | .090 | .885 | .090, .116, .124 | .141 |

| Prediction | Result |
|---|---|
| dogs: median eps_par above .5, KL above 1 | Held |
| novehicle: eps_par under .1 at s3, under .2 at s4, KL under .1 | Held |
| novehicle beats random controls by 1.5 times or more | FAILED. 1.22 times the mean of the seeds, and one seed ties it |
| Against strongest channels it is close | It beats them by 1.57 times, more than I expected |

# Round 2: the one-layer check (written before `one_layer_check.py` ran, 2026-11-03)

Plan `im-looking-at-the-playful-whisper.md`, Step 1. Shonu: "i don't accept the network isn't clean, id rather guess the map is wrong". So a miss below blames the setup, and it names the next suspect.

Lists are chosen on reading photos. Numbers are on sealed photos. Children are at their own peak, on their 64 strongest photos.

| Item | Prediction |
|---|---|
| The sum of all direct inputs equals z | Float error. If not, the script stops and I fix the script |
| Old measure at stage 4, k = 64 | The old list (share of z^2 changed .77) beats a random 64 |
| Stage 4 block 1: inputs to reach 50% of the summed effect, one list per child | 30 or fewer of 512, median |
| The same, one list per bundle | 15 or fewer |
| The same, one list per photo | 15 or fewer. This one is in-sample, and it only describes a photo |
| 30 random stage 4 channels | The same lines hold. They are not chosen by agents, so they are the fair sample |
| Inner channels: inputs from the block's stream to reach 50% | 30 or fewer of 512 |
| Stage 3 block 1, with the skip counted as one input | 5 or fewer, since the skip is about .77 of z |
| Share of z^2 changed at k = 64, one layer | Under .3, against .77 across four convolutions |

If stage 4 stays dense at one layer, the next suspects are position inside the 3 x 3 window, then near-twin channels that act as one parent.

## Round 2 results: the one-layer check (2026-11-03; `one_layer_check.json`, `one_layer_check.log`; sealed photos; validation crops, not N's training photos)

Lists were chosen on reading photos. Bundles are k-means, k = 3, inside one child. The per-photo count is in-sample.

| Item | Prediction | Result | Verdict |
|---|---|---|---|
| The sum of all direct inputs equals z | float error | stops above 1e-6 of z; never stopped | Held |
| Old measure, stage 4, k = 64: old list against random 64 | old is better | .770 against .978. 33 of 38 children | Held |
| s4 block 1: inputs for 50% of the summed effect, one list | 30 or fewer of 513 | 76 (range 45 to 96) | FAILED |
| The same, one list per bundle | 15 or fewer | 74 | FAILED. Bundles of this kind barely help |
| The same, one list per photo | 15 or fewer | 33 | FAILED |
| 30 random stage 4 channels | the same lines hold | 79, 78, 35. Same as the slice, so the slice is a fair sample on this measure | Same misses |
| Inner channels from the block's stream, 50% of the effect | 30 or fewer of 512 | s4 104, s3 36 | FAILED |
| s3 block 1, skip as one input | 5 or fewer | 8 | FAILED, narrowly |
| Share of z^2 changed at k = 64, one layer | under .3, against .77 | .036 | Held, far beyond |

| Share of z^2 changed, median | k = 4 | k = 16 | k = 64 |
|---|---|---|---|
| Old setup: s3 feeders of s4 block 1, four convolutions | .968 | .907 | .770 |
| Same setup, random list | .992 | .996 | .978 |
| s4 block 1 from its direct inputs | .428 | .151 | .036 |
| Same layer, random list | .950 | .916 | .755 |
| s3 block 1 from its direct inputs | .070 | .029 | .009 |
| s4 block 1 inner from s4 block 0, over the change from the mean | .635 | .385 | .146 |
| s3 block 1 inner from s3 block 0, over the change from the mean | .449 | .165 | .028 |

Reading, said plainly:

1. Distance was a large part of the old number. At one convolution, 16 inputs of 513 leave .151, where the old setup left .907. This does not compare like with like: the old setup kept 64 of 256 and crossed three ReLUs.
2. .151 of z^2 is a root mean square error of about .39 of z. So 16 inputs are close, and they are not the whole child.
3. The two measures disagree. It takes 76 inputs to reach half of the summed |effect|, yet 64 inputs leave only .036 of z^2. So the many small inputs largely cancel at the peak. I did not predict this. It fits the essay's "built from votes", and I have not shown that this is why.
4. Three-way bundles did not lower the count. So defect 4 in the plan has no support from this measure.
5. Stage 4's inner channels are the least clear layer measured. That is the next lead in the map's tools, with position inside the 3 x 3 window as the first suspect.

# Round 2, second check (written before `one_layer_map.py` and `reference_convnext.py` ran, 2026-11-03)

| Item | Prediction |
|---|---|
| Every layer of ResNet-18, k = 16, over the change from the mean | A stream layer is clearer than the inner layer below it, at every stage. Stages 1 to 3 are clearer than stage 4 |
| Block 0 inner, which reads the stage below | Less clear than block 1 inner at stage 4 (above .385 at k = 16) |
| Position inside the 3 x 3 window, as (channel, offset) inputs, s4 block 1 inner, k = 64 | About .15, no better than channel lists by much. Finer inputs are a harder count, so a large gain would be strong evidence |
| ConvNeXt-Tiny, last block: a stream direction from its 3072 neurons | k = 64 below .05. A reference only |
| ConvNeXt-Tiny, last block: a neuron from its 768 stream directions | Inputs for 50% at or above 100 of 768, since the stream has no privileged basis. A reference only |
