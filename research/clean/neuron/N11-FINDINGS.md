# N11 findings: the coarse wheel (2026-11-21)

Page with the photo sheets: https://claude.ai/artifact/Tn9Y45A91fGdzyZVzmsmWo
Full tables and method: `research/clean/NEURON-LOG.md`, section N11.

## The result in one picture

```
A = ResNet-18 (full)
  parents (stage-3 channels)        children (stage-4 block-0 neurons)
  s3:233  car wheel rim        ──►  n79 spoked cart wheel, n37 alloy rim, n458 wheel arch
  s3:69   radial, dark centre  ──►  n81 radial spokes, n297 rim and dial, n496 dark wheel or eye, n103 thin spokes
  s3:88   circular rim         ──►  n81, n103, n303 bottom wheel
  s3:158  dark curved edge     ──►  n303
  s3:120  eye ring             ──►  n496

B = the same network, stage 4 at a quarter of the width, trained by KL to A
  the same 5 parents           ──►  ONE wheel neuron, B n122: any wheel, plus a dome window and a cat's iris
```

## Evidence

| Claim | Number |
|---|---|
| B n122 covers what A splits | its top 200 photos lie inside the union of the 8 A wheel neurons' top 200s for .52 of them; 8 random A neurons .12; best single A neuron .20 |
| Children from A alone predict it (corrected) | union shares (.49 for s3:88's children) compare a ~1,300-photo union with B's 200 photos, so they only beat random unions of the same size (.12). The size-fair test is the step-3 rule below |
| Less room gives a coarser wheel | k = 512: spoked wheels (leans on A n81, .31); k = 128: any wheel (spread, max .20); k = 32: lines radiating from a centre (wheels, parachute lines, spider legs) |
| One parent, two coarse neurons | s3:69 feeds B's wheel neuron and B's shaggy-face neuron (n105); A splits it into wheel kinds and face kinds |

## How it was found

1. A = ResNet-18, because its stage 4 holds the whole "which car" split (removed: chance .091), unlike ConvNeXt-Base, which spreads it over about 12 blocks.
2. B was trained narrow from random on the cluster with fresh crops each step (`coarse/train18c.py`): k = 32 44.1%, k = 128 54.7%, k = 512 59.0% top-1 (A 67.7%). Seeds repeat.
3. Whole-photo sheets hid every job. Crops centred on each neuron's peak showed them (`coarse/sheet18.py`).
4. B reads A's frozen stage-3 channels, so an exact split of each neuron's input into those channels names the shared parents (`coarse/path18.py`).


## Later the same day: steps 3 and 4

| Question | Answer | Evidence |
|---|---|---|
| Rule from A alone (corrected after review) | **B's neurons match merged groups of co-firing A neurons better than single A neurons, more so as B narrows. A parent's children are NOT better than any co-firing group.** | best match .42-.45 for co-firing groups and .42 / .45 for a parent's 16 children (k128 / k32), single A .36 / .39; the group gain is +.03-.04 at full width, +.06-.09 at k128. Parent's children beat co-firing neighbour groups in only 31-38% of B's neurons |
| Q1, the read in weights | narrow B's reads are summed children's reads | cosine with a parent's 16 children summed .234 (k32), .241 (k128); single A .206 / .225; full width k512 ties (.242 vs .239) |
| Q2, the write | it writes the children's shared part | B n122's write: +.28 with the children's mean write; only .08 of it lies in their 7-dimensional difference space (random .007) |
| Seeds | both seeds hold the same tree at different levels | seed 0 split the radial parent into wheel (n122) and face (n105); seed 1's n101 stays at the parent level (.51 with both families) |
| Q5, growth (corrected) | each neuron sharpens the lean it had at initialisation | n122's weak face lean (+.17) goes to -.06 while its wheel match rises +.21 -> +.46; n52 was already a face neuron at step 1 (+.42) and strengthens (+.61). The early crops are of a near-random neuron, so "starts at the parent's job" is not shown |
| Step 4, theory check | **not tested at this scale** | with 16 of 512 neurons changed in one block, no method's errors line up (alignment 1.00-1.02); the built R (no fit) keeps 37-42% of the deleted job but loses locally to least squares (.0291 vs .0098 KL on wheel photos); its firing rate is 66% of positions, against 48% for B n122 (each A child 12-23%); scored per photo against n122 it gives .36 (mean of children's activations .46) |

## Q3: B loses fine splits as it narrows

| Network | Cars (chance .091) | Wheel classes (chance .034) | Face classes (chance .021) |
|---|---|---|---|
| A | .789 | .824 | .804 |
| B k512 | .679 | .717 | .741 |
| B k128 | .656 | .684 | .719 |
| B k32 | .512 | .601 | .628 |

(accuracy among the group's own classes; the full-width control already loses .06-.11, so part is the training budget)

## Open

1. The theory test needs many groups changed at once: coarsen A's whole stage 4 by the rule (every parent's children -> one neuron) and compare its error geometry with the least-squares build and with B, at the same width.
2. Build R from readings: threshold near B's rate (about half the positions), and families from co-firing, since the parent -> children tool gives families that barely fire together (tool gap: the edge needs the parent's photos).
3. Q3 (do B's readers still ask for the fine wheel split) and the negative poles of the parents.

## Multi-level work (later the same evening)

| Item | Result |
|---|---|
| Families for every neuron (A only) | 128 per block, by co-firing and by shared feeders. Wheel kinds are siblings: they share feeders (2 families) but rarely fire together (5-7 co-firing families) |
| One-level trained reference | block 0 at 128, block 1 = A: KL .236, top-1 63.8% |
| Errors at layer scale | changing all 128 families at once, their errors line up (added / singles 2.2-2.8), for deletion and the rule, like N6's 2.7 across blocks |
| Readers (one level vs two) | training keeps the error geometry (cos -.19 -> -.23, spread .99 -> 1.03); least squares does not (-.15 -> -.69, 1.03 -> .73) |
| Coarse neuron in weights | the plain sum of members fires at 9% (B: 42%) and the layer collapses (KL 2.5). With the threshold at B's rate: KL 1.37 (LS .51, trained .24). The OR of members (read kept intact) still costs 1.43, so **the missing piece is the write**: a coarse neuron does not write its members' average (B n122: cosine .28) |

Next: read what B's coarse neurons write, build the write rule from that reading, then the two-level build again.

## What a coarse neuron writes

| Readers | What the coarse layer writes | Evidence |
|---|---|---|
| A's fine block 1, fixed (T0) | its best guess of what A's full layer would have written, given its coarse firing, plus one loud root-level axis (busy texture against plain sky, 46% of its output variance) | cosine .63 with the copy; swapping in the pure copy costs .085 KL (.237 -> .322) |
| its own block 1, trained with it (B, at k128 and at full width) | a private message, 6-7.5 times A's size, three quarters outside A's main directions | copying collapses B (KL .69 -> 4.39; k512 .48 -> 3.54); same at full width, so it comes from co-training, not coarseness |

One-level gap (LS .512 to trained .237): reads .19, writes beyond the copy .085. Multi-level: each level's writes must be built with the next level's reads; copying each level compounds (two levels: LS 1.97, trained .69).
