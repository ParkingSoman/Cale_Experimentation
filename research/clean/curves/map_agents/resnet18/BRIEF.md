# Brief for a ResNet-18 map agent (test slice, 2026-11-03)

Your job: read part of a trained network the way the essay reads networks. Find which channels are parents of which, and write it down so that code and a blind reviewer can test it. You are not building an algorithm. Nobody has made one that works. Judgment is the point.

Shonu, who leads this project: "we need to have actually good maps" and "when we've created bad maps before, and you're going to see, oh, it's not possible to create a good map. No, it is possible. We just need to do a better job of it."

## The picture to test

A tree of channels. A parent channel in an earlier stage is broad. Its children in the next stage build on it and split it finer. A gate sits in front: cars and dogs may share a pose early, then a channel lets cars through and turns dogs away with its negative side. A smaller network stops at the coarser splits.

You test this picture. You do not force it. A channel with no real parent is a useful finding.

## The network

torchvision ResNet-18, on ImageNet. Read `map_agents/resnet18/net.py` for the names.

| Part | Channels | Grid | Note |
|---|---|---|---|
| stem s0 | 64 | 56 x 56 | Feeds stage 1's skip with no change |
| stage s1 | 64 | 56 x 56 | |
| stage s2 | 128 | 28 x 28 | Block 0 reads s1 through a 1 x 1 downsample and a residual path |
| stage s3 | 256 | 14 x 14 | The same |
| stage s4 | 512 | 7 x 7 | Then a mean over positions and one linear layer to 1000 logits |

Each stage has 2 blocks. Block 1 adds to the same channel numbers through its skip. So `s4:179` is one stream channel through the stage. The real branching is at the boundary between stages. `parents` looks through the skip, so use it instead of asking what the block before feeds.

The photos are 35000 ImageNet validation crops, 35 per class. They are not N's training photos. **Do not read anything under `cache/map18/sealed`.** Those photos score your work.

## How to read

Read these parts of `map_agents/essay.md` first: lines 33-35, 69-105, 225-251, 399-403, 465-482, 501-507.

1. A channel is a verb, not a list of nouns. It draws a distinction that later channels use. Ask: "if I could not use groups of nouns at all, how would I describe this?" The essay's "cats and cars" was a V against an O.
2. Look at both poles. `sheet --band neg` and `parents --band neg` show what is kept out.
3. Look at strong, medium and weak firings. A top sheet tests the center of a channel, not its breadth.
4. Look down at what builds it, and up at what reads it. They should line up.
5. Group firings by what feeds them (`by-photo`, `bundles`), not by the classes they land on. Offshoots that arrive by their own route are the essay's recursive generalization. They are not a defect.
6. Early layers are dials and textures. Do not expect objects there. Look for what the later channels use from them.
7. Crossword: each piece constrains the next. Read `map_agents/resnet18/NOTES.md` before each new channel. Append what you learn, in short lines with your name.
8. A good explanation is specific enough to lose. Before you test a channel, write one prediction that could fail.

Warnings from the essay: agents lean toward noun groups, and toward calling a channel messy when the microscope is at fault. Do not write "messy", "polysemantic" or "no structure" without showing what you checked.

## Tools

Run from `research/clean/curves`:

    .venv/bin/python map_agents/resnet18/ev.py COMMAND ...

Run it with no command for the list: `sheet`, `bundles`, `edit`, `parents`, `siblings`, `readers`, `by-photo`, `job`, `trace`, `score`, `drop`, `acts`, `carry`, `classes`, `photos-of`. A `sheet` prints a PNG path. Open it with Read. The crop sizes come from the stage, so a stage 1 crop is small and looks like a texture.

Do not write a clustering or grouping program. You may write a small test script in `map_agents/resnet18/tests/`. Its first line must be `# serves: H-map`. Keep to about 7 tool calls per channel. Other agents run at the same time.

## Coordination

Before you study a channel, append it to `map_agents/resnet18/claimed.txt` as `s4:179 yourname`. Check that file first, and skip a channel that another agent holds. When your route needs that channel, use their card.

## What to produce

**1. One JSON line per channel** in `map_agents/resnet18/cards/<yourname>.jsonl`:

    {"channel": "s4:179",
     "distinction": "one sentence, with no list of nouns",
     "positive": "what the strong firings share",
     "negative": "what the negative pole shares, and what it keeps out",
     "parents": [{"channel": "s3:12", "gives": "what it contributes, in a few words", "sign": "+", "sure": 0.7}],
     "gate": "if this channel lets one thing through and turns a related thing away, say which. Otherwise null",
     "readers": "what reads it, and whether that matches the distinction",
     "offshoots": "what arrives by another route",
     "family": "a short name that does not use a class noun where a shared pattern exists",
     "prediction": "the one that could fail, written before the test",
     "prediction_held": true,
     "sure": 0.7,
     "unread": "what you could not explain"}

List 1 to 4 parents, or an empty list with the reason in `unread`. Each parent needs evidence beyond a large number: its own sheet should show why it would build this channel, and `acts` should show the child fall more than the controls. Say in `gives` which of these you checked. For a stage 1 channel, the parent is the stem channel.

**2. Your final message**, at most 150 words: which channels you read, what kind of tree you saw, where the picture failed, and what the tools could not show you. Tell the truth about the failures. They are as useful as the successes.
