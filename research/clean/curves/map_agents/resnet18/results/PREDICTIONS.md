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

## Round 2, second check: results (2026-11-03; `one_layer_map.json`, `reference_convnext.json`; 15 random channels per layer; ConvNeXt 30 random, 3000 photos per half; validation crops, not N's training photos)

| Layer | Inputs | For 50% of the effect | Left at k = 4 | k = 16 | k = 64 |
|---|---|---|---|---|---|
| s4b1 stream | 513 | 78 | .483 | .156 | .038 |
| s4b1 inner | 512 | 119 | .733 | .456 | .179 |
| s4b0 stream | 768 | 131 | .664 | .337 | .126 |
| s4b0 inner | 256 | 50 | .569 | .238 | .058 |
| s3b1 stream | 257 | 11 | .066 | .044 | .009 |
| s3b1 inner | 256 | 49 | .632 | .295 | .047 |
| s3b0 stream | 384 | 63 | .642 | .305 | .077 |
| s3b0 inner | 128 | 18 | .359 | .070 | .004 |
| s2b1 stream | 129 | 3 | .086 | .034 | .002 |
| s2b1 inner | 128 | 18 | .422 | .122 | .005 |
| s2b0 stream | 192 | 29 | .457 | .144 | .011 |
| s2b0 inner | 64 | 8 | .338 | .035 | 0 |
| s1b1 stream | 65 | 2 | .074 | .006 | 0 |
| s1b1 inner | 64 | 10 | .331 | .055 | 0 |
| s1b0 stream | 65 | 3 | .027 | .007 | 0 |
| s1b0 inner | 64 | 10 | .319 | .030 | 0 |
| ConvNeXt-Tiny last block: stream direction from its neurons | 3073 | 109 | .159 | .117 | .065 |
| ConvNeXt-Tiny last block: neuron from its stream directions | 768 | 156 | .799 | .595 | .267 |

"Left" is the share of the change from the mean that stays outside the list, over sealed photos.

| Prediction | Result | Verdict |
|---|---|---|
| A stream layer is clearer than the inner layer below it, every stage | Holds at block 1, all stages. At block 0 only at stage 1. Stages 2, 3, 4: the stream is less clear (.144, .305, .337 against .035, .070, .238). A block 0 stream reads the stage below through a 1 x 1 downsample | FAILED at block 0 |
| Stages 1 to 3 clearer than stage 4 | True layer by layer, except s3b1 inner (.295) against s4b0 inner (.238), where the input counts differ | Mostly held |
| s4b0 inner less clear than s4b1 inner | .238 against .456, with half as many inputs | FAILED |
| Position, s4b1 inner, k = 64: about .15 | .496 on (channel, offset) inputs, against .179 on channels. 64 offsets use the weights of about 7 channels, where the channel list leaves about .65. So position gives a modest gain per weight | No large gain, so this suspect is set aside |
| ConvNeXt: direction from neurons, k = 64 below .05 | .065 | FAILED, narrowly |
| ConvNeXt: neuron from directions, 50% at 100 or more | 156 | Held |

Reading, said plainly:

1. On this measure, ResNet-18 looks about as dense as ConvNeXt-Tiny, or less. So "76 inputs for half of the effect" does not separate them, and my prediction of 30 was too tight. This is not like for like: ConvNeXt's stream has no privileged basis, it has more units, and it has fewer photos here.
2. So this measure gives no reason to doubt the tree. It also does not prove the tree. Agents and sealed tests must do that.
3. The lower stages are much clearer than stage 4 at one layer. The old blind test scored them worst (s1 13%, s2 20%, s3 29%), so that test is the first suspect there.

## Old cards on the new crop test (2026-11-03; `review2/old/`, sealed photos, two blind reviewers; stages 1 to 3 only)

No number was predicted before this run. The plan says: if they pass, the test was the problem.

| Stage | Cards | Old class test, share of channels at gap .2 | New crop test, share at gap .2 | Mean gap |
|---|---|---|---|---|
| s1 | 15 | 13% | 27% | -.11 |
| s2 | 11 | 20% | 64% | .27 |
| s3 | 31 | 29% | 48% | .13 |
| All | 57 | | 46% (line 85%) | .09 |
| Wrong cards | 9 | | | -.11 (line under .05) |

Chance for one channel to reach gap .2 is about 28%. So s2 and s3 sit above chance, and s1 sits at chance. The wrong cards score low, so the test can tell cards apart.

Reading: the old cards FAIL the new test as well. So the old test was only part of the problem. It scored s2 and s3 lower than the new test does. The cards themselves are not specific enough, above all at stage 1. Nine wrong cards is a small control.

# Round 2: lines and predictions for the new cards (fixed before any new card exists, 2026-11-03)

**Scope.** Per-block nodes multiply the slice, and 2.5M tokens end inside stage 4. So this wave reads stage 4's four layers, with at most 140 nodes. It says nothing about stages 1 to 3, except for links into `s3b1`. Stage 3 and below are a later wave.

**Two instruments changed before use, said plainly.**

1. "Parents carry" as written asked named inputs to reach 50% of the summed |effect|. The one-layer check shows that takes about 76 inputs, since small votes cancel. No card naming a few inputs could pass, whatever its quality. New measure: on sealed photos, the share of the change from the mean explained when only the named inputs stay real. A child passes when that share is at least .25 and at least 3 times a random list of the same size. Line: 80% of children, unchanged. On this measure the top 4 by effect explain .52 at s4b1 and .27 at s4b1 inner.
2. The crop test moves from 12 to 24 crops. Gaps came in steps of 1/3, and chance per channel was .28. It is now .11. The line stays 85% at gap .2.

**Stage 1's test is a suspect, and this is decided now.** A mean gap of -.11 at chance level fits "cards not specific". It also fits a test that cannot score a dial: a graded stroke-width description has no clear fire or not-fire crop. So when a later wave reaches stage 1, its test is a synthetic battery: images drawn with fixed stroke, width, angle and frequency, the card predicts an ordering, and code scores the ordering. The crop test stays as a second number there.

| Line | Pass | My prediction, stage 4 |
|---|---|---|
| Tree cover: the main parent moves the child by at least 2 times matched controls (root mean square, one layer, sealed) | 90% of nodes | 85%. It may miss |
| Parents act: the same, on all named inputs | 90% of links | 90% |
| Parents carry (above) | 80% of children | 60%. It may fail |
| Specific enough to lose: 24-crop test, gap .2 | 85% of nodes | s4b1 75%, inner 60%, overall 65%. It may fail |
| Real parents: pick the true parent of 5 from words alone | 70% | 60%. It may fail |
| Wrong cards lose | mean gap under .05 | Holds |
| Shonu's read | his | Not mine to predict |

I checked by eye 2 strong inner channels, which read at once, and 4 random ones, which fire weakly and did not read at a glance. So I expect the weak channels to set these numbers.

## Round 2: results for the new cards (2026-11-03; `cards2/`, `cards2_scores.json`, `review2/`; sealed photos; validation crops, not N's training photos)

Four agents read 135 distinct nodes (126 in stage 4, 9 named in s3b1). About 0.73M tokens for the reading. About 1.1M more for all blind tests. Stage 4 only, unless a row says otherwise.

| Line | Pass | Prediction | Result | Verdict |
|---|---|---|---|---|
| Tree cover | 90% | 85% | 88%. s4b1 100%, s4b0 92%, s4b0i 83%, s4b1i 77%. 15 of 126 have no direct main parent | Misses narrowly |
| Parents act | 90% | 90% | 100% of 369 links | Passes, and the line is too easy (below) |
| Parents carry | 80% | 60% | 67%. s4b1 93%, s4b0i 67%, s4b0 60%, s4b1i 47% | FAILS |
| Specific enough to lose: 24-crop test, gap .2 | 85% | 75% s4b1, 60% inner, 65% overall | 62% of 104. s4b1 70%, s4b0 70%, s4b0i 62%, s4b1i 50%. Chance about 11% | FAILS. Prediction about right |
| Real parents, 1 of 5 from words | 70% | 60% | 79% of 82 | Passes, above the prediction |
| Wrong cards lose | mean gap under .05 | holds | -.11 on 22 at stage 4. -.05 on all 25, with 3 in s3b1 at +.33 | Holds |
| Shonu's read | his | | waiting | |

Checks added after the first scoring, said plainly. None of them moves a line.

| Check | Why | Result |
|---|---|---|
| Harder parents test: all candidates by one agent, node names removed | The first test could be won on writing style. 75 of 82 true parents shared the child's agent | 85% of 66 |
| Independent parents test: a fresh agent describes 34 parents from sheets alone, seeing no card | An author could echo the child's words in its parent's card | 85% of 34, chance 20%. Parents in s4b0 14 of 14, parents in s4b1i 15 of 20 |
| Main parent's rank by effect | "Parents act" passed everything | Rank 1 on 80%, top 5 on 100%. Named lists explain .36, code's top list of the same size .38 |
| Careful retest of 30 crop items | I suspected reviewer noise after one bad item | Pass 70% against 67%. Correlation .83, same verdict on 90%. So my suspicion was mostly wrong. Both reviewers share a model, so they may share a bias |
| Old stage 4 cards on the 24-crop test | A before and after | Old 76% of 37. On the same 30 channels at s4b1: old 77%, new 70%. New better on 9, same on 9, worse on 12 |
| Crop test by strength, largest firing, split at the layer's median | | Strong half against weak half: s4b1 76 / 62, s4b0 90 / 50, s4b0i 71 / 50, s4b1i 74 / 26 |
| Crop test by the agent's own confidence | | Sure at .6 or more: 80% of 20. Under .6: 57% of 84 |

Reading, said plainly:

1. The links got better. Real parents went from 57% to 79 to 85%, and they hold with independent words. These are links one convolution apart, so they are easier than the old links, which crossed four. This does not compare like with like.
2. The words did not get better at the top layer. On the same channels the new cards are no better than the old ones.
3. The numeric lines cannot show that judgment added anything: the agents mostly named what the tool ranks first. The independent test says that those links mean something. It does not say the agent found them.
4. The shortfall sits in weak channels. That is the next lead in the map, the tools or the test. It is not a claim about the network.
5. Selection: the independent test covers strong links with cards, so it may flatter the tree. The fire crops sit near half of the largest firing (median .57 of it), so the crop test scores the edge of what a channel does more than its centre. I fixed that before the run, and I leave it unchanged.
