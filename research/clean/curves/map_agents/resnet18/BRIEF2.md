# Brief for a ResNet-18 map agent, round 2

Your job: read part of a trained network the way the essay reads networks. Find what each node grew from, and write it down so that code and a blind reviewer can test it. You are not building an algorithm. Judgment is the point.

Shonu, who leads this project: "if we aren't getting a clean map of the network we are doing it wrong ... i don't accept the network isn't clean, id rather guess the map is wrong".

So a node you cannot read is a lead. It points at a tool, an explanation or a motif that we do not have yet. Say what you checked, and what you would try next. Do not write "messy", "polysemantic", "no structure" or "many weak parents". The essay: "whenever you see polysemanticity, look harder". It is also fine to say you do not know.

## What was wrong last round

The old map joined two blocks under one name, skipped the inner channels, and named parents four convolutions below the child. The new tools look one convolution at a time.

## The network

torchvision ResNet-18, on ImageNet. A node is a channel at one layer.

| Node | What it is | Reads |
|---|---|---|
| `s4b1:179` | stream of stage 4, block 1. h = relu(z), z = r(inner) + skip | the inner channels of its block, and the skip |
| `s4b1i:179` | inner channel of stage 4, block 1. relu(a), with a one convolution from the layer below | the stream of the layer below |
| `s0:12` | the stem | pixels |

| Stage | Channels | Grid | Skip's share of z at a peak, block 1 |
|---|---|---|---|
| s1 | 64 | 56 x 56 | .71 |
| s2 | 128 | 28 x 28 | .70 |
| s3 | 256 | 14 x 14 | .77 |
| s4 | 512 | 7 x 7 | .07, so block 1 does nearly all the work here |

The order from the logits down: `s4b1` <- `s4b1i` <- `s4b0` <- `s4b0i` <- `s3b1` <- `s3b1i` <- `s3b0` and so on. Block 0 of a stage reads the block 1 stream of the stage below. A number is not a family: `s4b0:179` and `s4b1:179` can do different jobs, and `s4b1i:179` is a different channel again.

The photos are ImageNet validation crops, 35 per class. They are not N's training photos. **Do not read anything under `cache/map18/sealed`.** Those photos score your work.

## How to read

Read these parts of `map_agents/essay.md` first: lines 33-35, 69-105, 225-251, 327, 399-403, 465-482, 501-507, 569-573.

1. A channel is a verb. It draws a distinction that later channels use. Ask: "if I could not use groups of nouns at all, how would I describe this?" The essay's "cats and cars" was a V against an O. Look for what the strangest members share. The strangest member is the best clue.
2. Look at both poles. The negative side says what this point in the circuit keeps out.
3. Use the rings of `deep`. They are shares of the largest firing, so they show the channel's own shape. Some channels are dials, and the middle matters. Some are detectors, and it drops off a cliff.
4. Look down at what builds it, and up at what reads it. They should line up.
5. Look at the bundles. A child can have several routes, and each route is its own clue. An offshoot that arrives by its own route is recursive generalization, not a defect.
6. A neuron is built from votes. "One cat detector firing does little on its own, since many work together." So a small share for one input is normal. Ask which input this child grew from and what the group of inputs share. Do not ask which input carries it alone.
7. Early layers are dials and textures. Do not expect objects there. Draw a test image.
8. Crossword: each piece constrains the next. Read `map_agents/resnet18/NOTES2.md` before each new node. Append what you learn, in short lines with your name. A motif that you find may unlock other nodes.
9. A good explanation is specific enough to lose. Before a test, write one prediction that could fail. Cheap tests that work: images you draw with PIL (lines at angles, stripes, spokes, colour patches), `probe`, `edit`, class subsets.

Warnings from the essay: agents lean toward noun groups. Feature visualization can be reward hacked in late layers, so compare the free view with the natural view, and trust real photos when they differ.

## Tools

Run from `research/clean/curves`:

    .venv/bin/python map_agents/resnet18/ev2.py COMMAND ...

Run it with no command for the list: `deep`, `edit`, `probe`, `fv`, `inputs`, `readers`, `bundles`, `trace`, `acts`, `classes`, `photos-of`. `fv` is optional and not yet proven useful on this network: the one view I checked was faint. Do not spend a call on it unless real photos leave you stuck. A `deep` or `bundles` run prints a PNG path. Open it with Read. The inputs are exact. They sum to the value at the peak, and `acts` is an exact one-layer test.

Do not write a clustering or grouping program. You may write a small test script in `map_agents/resnet18/tests/`. Its first line must be `# serves: H-map`. Keep to about 7 tool calls per node. Other agents run at the same time.

## Your scope in this wave

This wave covers stage 4 only: `s4b1`, `s4b1i`, `s4b0`, `s4b0i`. A main parent of an `s4b0i` node lies in `s3b1`. Name it, and do not read onward into stage 3.

Start from your starting nodes. Work down: for each one, read its main parent, then that parent's main parent. Read at most 35 nodes, then stop. Reach all four layers. Do not read only the top layer.

Do not read the old cards in `cards/`, the old notes, or `results/`. They used the old setup and they would anchor you.

Strength varies a lot. Some inner channels fire at 4.0 and read at once. Others fire at 1.0 and do not read at a glance. A weak one is a lead too: look at its readers and its negative pole, and try a test image.

## Coordination

Before you study a node, append it to `map_agents/resnet18/claimed2.txt` as `s4b1:179 yourname`. Check that file first, and skip a node that another agent holds. When your route needs that node, use their card.

## What to produce

**1. One JSON line per node** in `map_agents/resnet18/cards2/<yourname>.jsonl`:

    {"node": "s4b1:179",
     "distinction": "one sentence, with no list of nouns",
     "positive": "what the strong firings share",
     "negative": "what the negative pole shares, and what it keeps out",
     "shape": "dial or detector: how the meaning changes down the rings",
     "main_parent": {"node": "s4b1i:191", "why": "why this child grew from it, and on which bundle", "sure": 0.7},
     "inputs": [{"node": "s4b1i:214", "type": "refines | part_of | pools | dial | relay | competitor | gate", "gives": "a few words", "sign": "+", "bundle": 2, "sure": 0.7}],
     "gate": "if it lets one thing through and turns a related thing away, say which. Otherwise null",
     "readers": "what reads it, and whether that matches the distinction",
     "offshoots": "what arrives by another route",
     "family": "a short name, with no class noun where a shared pattern exists",
     "prediction": "the one that could fail, written before the test",
     "prediction_held": true,
     "near_miss": "something that looks similar and should NOT fire it",
     "sure": 0.7,
     "unread": "what you could not explain, and what you would try next"}

| Input type | Meaning |
|---|---|
| refines | a finer version of the same pattern |
| part_of | a part that this whole is built from |
| pools | one of several variants that this node treats as the same |
| dial | a graded signal, such as colour or size |
| relay | a near-twin that passes the job along |
| competitor | a sibling that argues against this node |
| gate | it lets this route through or keeps something out |

`main_parent` is the input this node grew from: your best judgment of its one main parent. Your tree is the set of main parents. The other inputs are real edges and they do not count against the tree. A main parent needs evidence beyond a large number: its own view should show why it would build this node, and `acts` should show the child moves more than the controls. If you truly have no main parent, write `null` and say what you tried in `unread`.

**2. Your final message**, at most 150 words: which nodes you read, what motifs you found, where the reading stopped and why, and what the tools could not show you. Tell the truth about the failures. They point at the next fix.
