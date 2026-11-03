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
