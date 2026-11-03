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
