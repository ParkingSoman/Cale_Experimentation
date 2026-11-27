# Neuron study: tiny sessions

One question per session. One entry per session: the question, the objects (picked by a rule written first), the result, and the debrief. The next session starts from the last debrief.

Level for now (Shonu, 2026-11-27): real neurons only. The tree structure is not touched.

| File | Holds |
|---|---|
| `read3.py` | The reading tool: crops by band on both poles, intersection images, votes (the read), the readers of the write, and judge sheets |
| `cellpos.py` | Where in the 7 x 7 grid a neuron fires |
| `score_judge.py` | Scores the blind judges against the key |
| `out/<session>/` | Sheets and numbers (git-ignored). `_key/` is never shown to readers or judges |

How a session runs:

1. Name the objects, and the rule that picked each one.
2. Run `read3.py`: the same tool on every neuron.
3. Blind readers, all opus Explore agents. One reads the positive poles. One reads the negative poles and never sees the positive side. One reads the writes. They see labels only.
4. Read one essay passage in full.
5. Compare the neurons as verbs, with numbers next to the words.
6. Propose ONE line for the read or the write. Talk about it before any build.
7. Debrief.

## Objects (shared by sessions 1-2)

| Name | Object | Picked by |
|---|---|---|
| X | A (ResNet-18), `layer4.0` inner ch 184 | The block-6 parent with the largest merged child group at 50% (page v2 cut) |
| Y | W50R_s0 (trained from random at 50%), `layer4.0` ch 139 | The highest firing correlation with X on 8,000 Broden photos (r .32; .31 on ImageNet) |
| R86, R219 | W50R_s0 `layer4.0` ch 86 and ch 219 | Y's two nearest rivals by the same correlation (.25, .21) |
| Z | Our page-v2 50% build: the `layer4.0` node that holds ch 184 (group [184, 300], index 93). Its read = ch 184's own read | Run on the BUILT network's own stream |

## Session 1 (2026-11-27): what do X, Y and Z do?

Tool: `scratchpad/vr_verbs.py` (now `read3.py`). Windows used mirror padding at the photo edges, which was a tool limit. 10,000 val photos.

| | X parent | Y trained | Z ours |
|---|---|---|---|
| Fires on | 20% of cells | 24% | 97% |
| Deepest negative | -2.18 | -3.26 | -0.43 |
| Positive verb | thin fine elements crowded into a centred patch, often radiating | small soft blob-type marks scattered over a region, no long straight lines | fine regular lines over the whole window, no centre |
| Keeps out | one compact rounded form seen front-on, split by a midline | hard closed rings, and rows of rounded units | barely a gate |

Reading: Y keeps X's distinction, widened (essay 225, 239, 251). Z has X's exact read, but the coarse layers before it changed what its inputs mean. Its gate is gone (essay 233).

Proposal (talked about, not built): the coarse read = the parent's distinction widened, with votes on the jobs of the coarse upstream nodes. The gate should close about as often as the parent's.

Debrief: Shonu approved this way of working. Open items: mirror padding, the write read only from mean images, whether Y is the right match, and no blind judge yet. These became session 2.

## Session 2 (2026-11-27): is the X / Y reading right, and what do the writes do?

Tool: `read3.py s2` (grey padding, crop sheets for the readers, rivals, judge sheets) + `cellpos.py`. 10,000 read photos (every 5th val photo). Judge photos: a different 10,000 (offset 1).

### 1. Padding fixed: Z's shift is real, and larger than a border effect

| | X | Y | Z |
|---|---|---|---|
| Mean pre-activation, middle cell | -0.32 | -0.36 | +0.22 |
| Fire rate, middle cells | .12-.15 | .19-.21 | .95-.97 |
| Fire rate, outer ring | .22-.55 | .23-.58 | 1.00 |
| Share of top-250 max cells on the outer ring (no position effect = .49) | .52 | .37 | .95 |

Z's whole map is moved up by about +0.5, and the ring moves up even more. X and Y have only the usual top-row lift.

### 2. The match: X's job is spread over three trained neurons

| Blind reader's answer | Closest to X | How it differs from X |
|---|---|---|
| Positive side | Y | Both mark fine, dense texture packed into a compact soft patch. Y is wider toward soft fluff and scattered speckles. Y drops X's hard regular repeats (spokes, chainmail, fins). Those fit R86 (regular periodic straight structure) and R219 (small regular units on a rounded surface) |
| Negative side | R219 | Both keep out a compact rounded form in the centre with dark eye-like spots. R219 needs dark dots on a pale smooth mass and goes deeper (-3.72 against -2.18). X's glossy radial discs go with Y's gate instead (hard concentric rims) |

| Neuron | Positive verb | Keeps out | Fires on | Negative range |
|---|---|---|---|---|
| X | fine, dense, repeated small-scale texture in a compact round patch, against smooth surface | one compact, rounded, symmetric form filling the window (a pair of dark spots facing the camera, or a glossy radial disc) | 20% | -2.18 to -1.21 |
| Y | soft, fine-grained, dense texture filling a soft-edged patch, against smooth surface | a hard circular rim, ring inside ring, and rows of small round or boxy units | 24% | -3.26 to -1.87 |
| R86 | regular periodic straight structure (bars, lattice, rows of characters) | thin dark pointed lines splayed out of a core | 32% | -2.87 to -1.77 |
| R219 | small regular repeated units wrapped over a rounded surface | small dark round spots inside a compact pale smooth mass | 25% | -3.72 to -2.10 |

Unexplained strong crops (reader gaps), out of 32 per sheet: positive X 1, Y 4, R86 2, R219 3. Negative X 5, Y 5, R86 8, R219 12.

Reading: the trained 50% network does not hold X in one neuron. It splits on a different line: soft against hard (Y = soft texture, with hard rings in its gate). X splits fine elements (soft or hard) against a front-on rounded form. Essay 99: "Our smallest model only has a rough version, with cars and dogs together but not by pose": a small model can group along a different, rougher line.

### 3. The writes, read from crops

| | Readers turned ON | Readers turned OFF | Write as one verb |
|---|---|---|---|
| X (+2.2 to +2.5 / -1.3 to -1.6) | smooth curved bare surface crossed by colour bands; hard straight-edged surface at the frame edge; pale rounded lit blob on a textured surround | bold two-tone patches; flat fan-shaped sheet of fine radiating veins; whole upright figure at small scale (reader gap: one crop) | toward close surfaces that fill the window, away from fine detail and whole small figures |
| Y (+1.0 to +1.5 / -1.0 to -1.5) | convex tapered profile (reader gap: one crop); bright flat panel with small high-contrast marks; front-on close surface of fine soft texture | tall upright forms with vertical folds, in rows; reader gap (no shared pattern found); one compact upright body with a rounded top | toward close surfaces that fill the window, away from whole upright forms at mid scale |

Reading: X and Y push the flow the same way on scale and layout (close, frame-filling surface against a whole figure seen from further away). They differ on texture. X's write is stronger. The session-1 write reading from mean images ("centre spot vs window-filling layouts") did not hold up once crops were read.

### 4. Blind judge (checklist step 10), noun-free verbs only, held photos

Share of the judge's 24 picks that are real targets (chance .50). Two judges, A and B.

| Sheet | Real words (A, B) | Control words (A, B) | Margin |
|---|---|---|---|
| X positive | .83, .79 | .67, .71 | +.12 |
| X negative | .79, .75 | .71, .71 | +.06 |
| Y positive | .71, .75 | .62, .58 | +.13 |
| Y negative | .79, .75 | .46, .54 | +.27 |

The real words beat the control words on all four sheets, for both judges. But the controls score high (.46-.71), so the test is weak. The control words were other texture words that partly fit (for example, "fluffy strands" fits the fur around front-on faces). The X negative verb (+.06) is not separated from its control. Fix next time: unrelated control words, and distractors from neurons with a similar texture.

### Debrief (2026-11-27)

| Item | State |
|---|---|
| Mirror seams | Fixed |
| Writes read with crops | Done. X and Y agree on scale and layout |
| Is Y the right match? | Only for the positive side. X's gate matches R219. X's hard repeats match R86 and R219 |
| Blind judge | Real beats control everywhere, but the controls are too close to the real words. X negative not separated |
| Z | Its whole map moved up by about +0.5. Upstream of the read, not in it |

Questions for session 3 (Shonu picks):
1. Does the trained network re-cut X's distinction (soft against hard), and does X's soft half map one to one onto Y? Read X's positive crops split by Y's firing.
2. Where does Z's +0.5 shift come from? Read its votes on the calculated stream against X's votes on A's stream, voter by voter.
3. Make the judge test fair (unrelated controls, harder distractors), then re-judge X negative.
