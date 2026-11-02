# Brief for a map agent (pilot, 2026-11-01)

Your job: read part of a trained network the way the essay reads networks, and write down what you find as claims that code
can check. You are not building an algorithm. Nobody has made one that works. Judgment is the point.

Shonu, who leads this project: "you need to have subagents figure that stuff out rather than trying to develop a one size
fits all sort of algorithm".

## What you read

| Item | Where |
|---|---|
| The essay "Neural Networks Are Already Clean" | `map_agents/essay.md`. Read lines 33-35, 69-105, 145-161, 225-251, 399-403, 465-482 first. Read more when stuck |
| The network | torchvision ResNet (your task names which one), last hidden layer: 512 channels (ResNet-18) or 2048 (ResNet-50), mean over 7 x 7, then one linear layer to 1000 logits |
| The photos | P: 20000 ImageNet validation crops, 20 per class. These are not training photos |

## How to read (from the essay; follow it, do not fall back on habit)

1. A channel is a verb, not a list of nouns. It draws a distinction that downstream readers use. Ask: "if I could not use groups
   of nouns at all, how would I describe this?" "Cats and cars" was really a V against an O.
2. Look at both poles. The negative side says what distinction matters here.
3. Look upstream at how it is built and downstream at what reads it. They should line up.
4. Group firings by the circuits that feed them (`feeders-by-photo`), not by the classes they land on. A channel's strong firings may
   be facets of one core plus offshoots that arrive by their own route. Offshoots are not a defect: they are the recursive
   generalization of the essay.
5. Look at strong and medium and weak firings. A blind top tests the center, not the breadth.
6. Crossword: each piece constrains the next. Read `NOTES-<net>.md` before you start and before each new channel. Append what you learn
   (short lines, your agent name). Others read it.
7. A good explanation is specific enough to lose. Before you test a channel, write one prediction that could fail.

Warning from the essay: agents are biased toward noun groups, and toward calling a channel messy when the microscope is at fault.

## Evidence tools

Run from `research/clean/curves`, with `/opt/homebrew/bin/python3.11 map_agents/evidence.py NET COMMAND ...`. NET is `resnet18` or
`resnet50`. Read the top of `map_agents/evidence.py` for the commands (sheet, readers, feeders, feeders-by-photo, carry, photos-of,
classes, ablate, corr, score). A sheet command prints a PNG path. Open it with the Read tool to see the crops. Level 0 is the last
hidden layer, level 1 is one block earlier, and so on.

Keep calls small: other agents run at the same time. You may write a small test script of your own in `map_agents/tests/`. Its first
line must be `# serves: H-map`. Do not use more than about 60 tool calls in all. Do not write a clustering or grouping program:
that is the thing we are moving away from.

## What to produce

**1. Cards**, in `map_agents/cards/<net>_<your name>.md`. One short entry per channel you studied: the distinction in one sentence with no
noun list, its poles, what feeds it, what reads it, your prediction and whether it held, offshoots, and how sure you are.

**2. Claims**, in `map_agents/claims/<net>_<your name>.json`. A claim says: this set of channels (a district) does its work on this region
of photos, and elsewhere N's readers get little from it. The region must be the network's own: prefer a gate, a read of channels with
a threshold. Format and meaning are at the top of `map_agents/claim_error.py`. Run it:

    /opt/homebrew/bin/python3.11 map_agents/claim_error.py NET map_agents/claims/<net>_<your name>.json

Your word never counts as an error of 0. The code decides. Report its numbers as they come, good or bad. A claim that fails is
useful. Try to make your best claim better by understanding why it failed. Never tune a threshold just to lower a number.

**3. What does not fit.** Our claim types say that a district is off outside its region, or quiet in what it writes there. ResNet-18's
channels are on for most classes, so these types may be the wrong description. If what you see fits no type, describe in plain words
what kind of statement would be true of these channels, how code could check it from N and P, and which channels show it. That is a
valuable result, as valuable as a good claim.

**4. Your final message** (at most 200 words): what kind of structure you found, your best claim's numbers, what does not fit, how many
channels you studied. Do not call anything messy or clean without showing what you checked.
