# Deriving the coarse neuron: report (2026-10-22, D-series)

Question (Shonu): stop fitting. Study real coarse neurons like a mathematician studies a circle. Derive what makes a
coarse neuron from its fine neurons. Training is an approximation of that equation, not its ceiling.
Log with all numbers and rules: `research/clean/VISION-LOG.md`, sections D1-D4. Code: `vision2/d1_*.py` to `d4_*.py`.

## What a coarse neuron is (observed in real small models and in a toy where the truth is known)

| Property | Evidence |
|---|---|
| It is a CONSENSUS of a group of fine neurons, not their OR | atto vs tiny, 80 random neurons: sum of parts .50-.55 > max .37-.43 > best part .33-.34; quiet when one part fires alone (last stage -.11 std), high when 3+ agree (+2.0 std) |
| One GELU of the summed, scaled votes expresses it | .55-.61 (vision), .91 (toy) |
| It fires about as often as its parts (rate law) | firing-rate ratio .88 (last stage), 1.43 (stage 3), 1.03 (toy); in the toy the best threshold is exactly the rate law |
| The group is found from the large model alone, by co-firing | parts co-fire .17-.19 vs .003 random; rank 20-28 of 23k around the top part |
| The group runs across siblings AND across depth | atto 10/320 (side of road vehicles) = tiny's minivan, station-wagon, sports-car specialists in blocks 15-17 |
| The map's one-parent tree does not hold it | 45 of 50 top parts are "other" in map v2 |

Caveat: two separately trained models share only a loose neuron basis (R^2 .13-.41 from the top-10 parts, the same from
nano as from tiny), so the small model is a noisy truth for single neurons.

## Toy (8 coarse x 8 fine classes; wide 256 vs narrow 16)

A 16-neuron net built from the wide net with NO training (co-firing groups, consensus read, rate-law threshold, Bayes
write) reaches coarse accuracy .93 vs trained .95 (fine .43 vs .69). Ideal pure-coarse reads reach coarse .997, above
training, but lose fine accuracy. The write must be the Bayes answer (expected wide probabilities given the coarse
activations); local expansions around the wide model fail because a coarse net cannot sit close to a fine one.
Global linear algebra without neuron groups collapses (coarse .34-.42): the neuron structure is what makes it work.

## ConvNeXt

| Result | Number |
|---|---|
| Inside one block, neurons barely co-fire (best partner median .27; 79% have a better partner in another block) | width merge fails: .122 vs cut .083 |
| A block's votes survive removal of its own filter (block B+1's view, conditional expectation) | R^2 .91-.96 |
| Same neuron budget in blocks 16-17 (3,072): thin both to 50% + training / GN (last session) | .224 / .217 |
| Derived depth merge (no training) | .162 (groups .206) |
| **Simply drop block 16, keep block 17 as is** | **.104** |
| Derived repairs of the drop (mean bias / restored reads / stream-MSE write) | .125 / .381 / .123, all worse |

Honest status: no derived ConvNeXt arm beats training at the RIGHT structure yet. The strongest finding is structural.
"End a block early" costs half as much as thinning two blocks and repairing them with training. That matches the essay
("ends early") and atto having fewer blocks. Linear derived corrections (stream MSE, conditional-expectation reads, mean
bias) hurt a clean drop. The readers' view and LayerNorm's scale are the likely missing pieces.

## D5: strong reference, and what training changed

Strong training reference at the right structure (drop block 16, train block 17, lr swept, early-stopped): sealed KL .0942
vs plain drop .1042 (10% won back). What training changed, read as numbers with meanings:
- It does not rebuild the lost parent votes (slope .002).
- It moves each neuron's average output 65% of the way back to its intact level (corr .81 over 3,072 neurons), through
  small read turns (cos .995), not thresholds.
- The change is spread out but reproducible across runs (cos .91-.98). No simple source explains it yet (parents, own
  vote, LayerNorm size: R^2 <= .04).
- Every one-number derived fix tried (level law, rate law, mean write, size restore, vote gain) is worse than the plain
  drop. The plain drop is a local optimum for simple fixes.

- The head averages block 17 over the 49 positions, so it only sees each neuron's per-image pooled output. Training
  moves that pooled output ~10% of the way back to intact (slope .10, per-neuron corr .33), matching its 10% KL gain.
  Forcing the full pooled restore in one linear step needs read changes larger than the reads (1.3x vs training's .1x)
  and hurts (.186).
- The trained change is reproducible across image orders (cos .93 read, .95 write; seed-7 run .0934).

- Pictures: block 16 supplied GATES. When it is dropped, a wolf neuron loses wolves (43 -> 6 % of max) and another
  neuron starts firing on dogs it ignored (0 -> 25). Training rebuilds those gates about 10%.

## D5b-D6: strong reference, and the self-healing law

- Reference check (Shonu: "suspicious it only recovers 10%"): training block 17 alone OVERFITS (train KL .027 vs held-back
  .104). With 90k images (unused + mirrored) and a feature loss: sealed .0836 = 20% won back, flattening. Top-1: intact
  78.4%, drop 76.2%, best reference 76.7%. Literature (PRACTISE, 2303.00972) recovers most of a block drop with 50-1,000
  images by training ALL remaining layers with feature mimicking; our one-block reference is far below that.
- SELF-HEALING LAW. Removing a late block shrinks the stream (r = .73); LayerNorm then reads every remaining vote ~1.4x
  louder. On each downstream neuron's own territory this replaces the lost gate: slope of the gate on (1-r)(final vote)
  1.07 (bootstrap 1.01-1.14; law says 1). Off territory the block added ~2x proportional inhibition, which the drop loses
  (false firing). Across 10 blocks LayerNorm covers .5-1.2 of the lost gate on territory. The mechanism is one shared direction: in the
  readers' metric 91% of the stream and 96% of block 16's write lie along the stream's common (mean) direction, and block
  16 adds ~one more copy of it (lambda 1.07; fluctuations alone: .17). Removing it turns LayerNorm's volume up. The
  novelty term separates stage 4 from stage 3 (corr .87 over 10 blocks) but does not rank blocks within stage 3 (.21).
- Every simple fix failed for an identified reason: restoring the stream size kills the healing; MSE estimates shrink
  variance (regression to the mean); per-neuron fixes, even an exact per-neuron quantile map, break the JOINT pattern
  across neurons; partial cuts kept the block's constant bias, which restores size without gates (bias alone doubles
  the damage: .205 vs .104). This bias confound also affected all earlier 50% cuts.
- Pictures: block 16's "volume" neurons fire on salient man-made objects with dogs on their negative pole; its gate is
  59% inhibition. The drop's damage sits on man-made objects out of context (a tie on a wall, saltshakers close up);
  94% of changed answers go to a neighbour class.
- Retracted: ".0037 from 7% of the write" and ".077 from the readers'-subspace novelty" (both circular: they rebuild
  the intact input, fully or in 702 of 768 directions). The second one only says the 66 least-read directions (the size
  axis), left at their dropped values, cost KL .077.

## Next (D7)
1. Compress the located information: can a small module (low-rank map of the stream, or a few block-16 neurons with the
   bias handled by the law) produce block 16's readers'-subspace novelty?
2. Strong reference with ALL remaining layers trainable (PRACTISE recipe) to know the true training ceiling.

## Next (older)

1. DONE (D5): strong reference .093-.094. Next: read the reproducible trained change neuron by neuron with pictures
   (top-changed neurons, images where their pooled output moved), before any new fix.
2. Which block to drop (15/16/17 and stage-3 blocks), and dropping several.
3. A derived repair that respects the readers' view and LayerNorm's scale (the toy's Bayes-write lesson, applied to the
   stream).
