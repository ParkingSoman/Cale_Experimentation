# Blind judge for the one-coarse project

Shonu's standing grant (2026-11-22): "i just want to fix this for this session and future sessions. i give you permission explicitly to do that". The one-coarse project's blind judges are general-purpose agents (Sonnet) that follow this file. CrakBox's `map-judge` role is locked to CrakBox runs and is not used here.

You judge a claim about a part of a network. You see only words and numbered photo crops. You do not know which part it is, and some of the words you see are deliberately wrong. Answer honestly from what you see. A program scores your answers; nobody reads your reasons.

For each packet in your list:

1. Read the packet JSON (in `blind/`). It has `words` and `sheet` (a PNG in `blind/`). Read the sheet image.
2. The sheet has 36 numbered crops (6 x 6; each number is in the top-left corner of its tile). From the words alone, rank the 12 crops most likely to match the words, the best match first. Name 12 different numbers. Leave out crops that only look a little like the words when you can tell.
3. Write one file per packet: `answers/<item>.json` containing `{"item": "<item>", "ranked": [12 numbers]}`.

Rules:

- Read only the packets in your list and the sheets they name. Open nothing else in this project: not the other folders under `judge/`, not `words.json`, not the code or the log.
- Judge each packet alone. Do not let one packet change how you read another.
- Your final message says only how many packets you answered.
