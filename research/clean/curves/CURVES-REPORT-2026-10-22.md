# Curves and loops: deriving a small layer from a large one (report, 2026-10-22)

Full numbers and every failed attempt: `research/clean/curves/CURVES-LOG.md`. Code: `research/clean/curves/`.

## Summary

1. We reproduced the essay's sentence in our own network: one coarse neuron that replaces the ten curve detectors
   becomes a "curves & loops" neuron (feature visualization: one loop; crops: wheels, lenses, dials).
2. The derivation alone does NOT beat training at the same size. A dev-set result said it did; on sealed images the
   derived three-neuron build is worse than three neurons trained from random, by .017 (paired se .004). Retracted.
3. As a STARTING POINT, the derivation is strong. Three neurons trained from the derived start beat ten neurons
   trained from random (paired difference -.0033, se .0011). Three neurons trained from random only reach the
   one-neuron level (paired difference with k=1: +.0002, se .0012).
4. Where the derived build loses: mostly the write (how the next layer reads the coarse neurons), not the read.
5. At one neuron, the derived coarse neuron beats keeping the best single fine neuron (paired -.052, se .007).
   Coarsening is not pruning.
6. A second family (the circles in 3b, which tile size instead of orientation) repeats findings 2 and 3 on sealed
   images: derived start beats random start by .021 (se .0014).

All rankings in this report are sealed, paired comparisons. Dev-set differences under about .015 are not claims:
every trained model scored about .02 lower on sealed than on dev, while the derived builds did not.

## 1. What we did, and how it differs from the essay

The essay crushed all of InceptionV1 layer 3b (480 neurons) to 10 and retrained the whole network from scratch. It saw
one "curves & loops" neuron where the big model has ten curve detectors.

We cannot retrain InceptionV1 from scratch on this Mac. We also showed that a whole-layer crush into the frozen later
layers is hopeless: even the best 10-dimensional linear copy of 3b gives KL 15.9 and 1% top-1.

So we crushed ONE FAMILY. The ten curve detectors (3b: 379, 406, 385, 343, 342, 388, 324, 340, 330, 349) become k coarse
neurons. Everything else in the network stays exactly as trained. The coarse neurons read the same input as the curve
neurons, so fine and coarse weights live in one basis. We reproduced the essay's SENTENCE ("ten curve detectors -> one
curves & loops neuron"), not its experiment.

Measure: KL from the intact network's answers (lower is better), and top-1. Removing the curve family costs KL .59-.68.
Data split per class: 35 train, 5 dev, 10 sealed. Sealed images were used once, in the cold test (section 6).

## 2. The essay's result appears in our network

One trained coarse neuron becomes a curves & loops neuron:
- Feature visualization: one loop of concentric curves. The ten fine neurons render as ten rotated curves.
  (`cache/fv-fine-*.png`, `cache/fv-L-k1-m379-lr3e-2-0.png`)
- Strongest natural crops: whole round things. Wheels, whitewall tires, hubcaps, camera lenses, dials, a stethoscope loop.
- On a centred circle of radius 20 px it gives about twice the response of the best fine curve neuron, peaking
  exactly at the centre.

## 3. What a coarse curve neuron is (measured, no fitting)

| Property | Measured |
|---|---|
| Fine family | ten rotated copies around the orientation circle; equal kernel norms; biases ~0; each on at 9-15% of spots; at one spot 1-2 are on, and they exclude each other |
| Readers (layer 4a) | they read the family mostly through its two lowest "waves" over orientation: the average h0 (20%) and one cycle h1 (39%, null 17%) |
| The coarse neuron | tracks the family with each member MOVED to its own centre of curvature, then added: correlation .52-.60 (plain sum .33-.40; reversed direction about 0) |
| Why that arrangement | moved to their centres of curvature, the members fire together (4 of 10 at the coarse neuron's strongest spots, against 1-2 at one spot). A single ReLU can express "several agree"; it cannot express "one of ten that exclude each other" |
| Two trained twins (different starts, same KL) | agree at .80 |

Mathematically, the coarse neuron is the family averaged over rotations about one point (a group average).

## 4. What training can and cannot find (sealed numbers)

| Reference | sealed KL |
|---|---|
| k=1 trained, either start | .121 |
| k=3 trained from random | .122 (the same as ONE neuron) |
| k=10 trained from random (the exact copy of the original sits at .000006) | .041 |
| k=3 derived, no training | .139 |
| **k=3 trained, starting from the derived three members** | **.037** |

- At k=3, random-start training ends where one neuron ends (paired difference +.0002, se .0012). At k=10 it does
  much better (.041), but it still does not get back to the original (.041 against .000006).
- Starting from the derivation, three neurons (.037) beat ten neurons trained from random (.041): paired difference
  -.0033 (se .0011). A small margin, but three neurons against ten.
- RETRACTION: on the dev set the derived k=3 looked better than random-start training (.133 against .139). On sealed
  images it is worse by .017 (paired se .004). The dev gap was noise. The derivation alone does not beat training at
  the same size here. Derivation followed by training does, by a wide margin.

## 5. The derived equation, in words

Nothing here is fitted to the network's output. Every number comes from the large model's weights or from image
statistics on training images.

**One coarse neuron (k = 1): the curves & loops neuron.**
1. True quantity: add up the ten fine curve neurons (after their ReLU), each one read at the spot where its curve would
   sit on a loop around this point. This is the family averaged over rotations about one point.
2. Read: the best linear read of that quantity from the coarse neuron's own input window (the whitened spike-triggered
   average). Bias: its own intercept.
3. Write (into layer 4a): for each reader, the best estimate of what it used to get from the family, rescaled to keep
   the reader's spread: column = Bayes column / |rho|. Reading two trained neurons like a best-fit line gives the same
   form (exponent -0.80 and -0.89, about -1).

| Constant | Value | Meaning |
|---|---|---|
| direction of each move | toward the member's own centre of curvature | the only direction that works; the reverse gives correlation ~0 |
| loop radius r | 1 position (8 px) in a 7x7 window, chosen on dev from {0, 1, 4} | we think it is set by the window size, an ARCHITECTURE constant, not a family constant. The fine curves prefer 50-75 px arcs; the trained coarse neuron prefers 25 px circles; a co-firing rule (pre-registered r = 4) did worse |
| write exponent | -1 | keep the reader's spread; read off training as -0.8 to -0.9 |
| write prefactor | about 0.7 in training | NOT explained. Two noisy fits (.68, .75); using 0.7 made results worse |

**k >= 2: keep members, spaced evenly.**
1. Keep k of the ten members, spaced as evenly as possible around the orientation circle. Three at 120 degrees is the
   smallest set that carries the average (h0) and both parts of the one-cycle wave (h1) that the readers use most.
2. Write the Bayes estimate of the removed members' input into the readers (a regression from the kept channels).

The k = 1 recipe (a new coarse neuron) and the k >= 2 recipe (kept members) are NOT unified yet.

## 6. Cold test (sealed 10k images, one run)

Intact top-1 65.8%. Full table: `CURVES-LOG.md`, section "COLD TEST". Selection is labelled.

| Build | sealed KL (se) | top-1 | how it was chosen |
|---|---|---|---|
| family removed | .684 (.019) | 62.5% | - |
| constant fill only | .392 (.014) | 64.1% | - |
| keep the member with the most variance | .325 (.012) | 64.6% | baseline |
| D1: sum of moved kernels | .375 (.013) | 64.2% | pre-registered, FAILED |
| derived loop neuron, k=1 | .250 (.009) | 64.9% | r and write chosen on dev from 9 variants |
| derived k=2 | .220 (.008) | 65.0% | |
| derived k=3, 10/120/250 deg, Bayes | .139 (.005) | 65.6% | first pick |
| derived k=3, 50/170/290 deg, variance write | .128 (.005) | 65.6% | control triplet |
| derived k=3, naive (3 largest-variance members), Bayes | .172 (.007) | 65.1% | baseline |
| derived k=5, variance write | .074 (.003) | 65.9% | |
| trained k=1 | .121 (.004) | 65.7% | reference |
| trained k=3, random start | .122 (.005) | 65.6% | reference |
| trained k=3, derived start | .037 (.001) | 65.8% | reference |
| trained k=10, random start | .041 (.002) | 65.8% | harness |

Paired differences (same images): evenly spaced vs naive triplet -.033 (.005): spacing matters. 10/120/250 vs
50/170/290 with Bayes -.012 (.005); with the variance write the order flips; NOT separated overall. Bayes vs variance
write at k=3: not separated. Derived loop neuron vs keeping one member: -.052 (.007): the coarse neuron beats pruning.
Derived loop neuron vs trained k=1: +.128 (.007): training is much better at k=1.


## 7. Read and write: where the derived build loses

We used training as a measuring tool: freeze the derived read, train only the write.

| k | derived (no training) | derived read + trained write | fully trained (best start) | read costs | write costs |
|---|---|---|---|---|---|
| 1 | .242 | .160 | .140 | ~.02 | ~.08 |
| 3 | .133 | .078 | .042 | ~.035 | ~.055 |

(dev KL, 1000 images. Sealed for k=3: derived .139, trained write only .076, trained .037.) The write is the bigger gap in both cases.

How the k = 1 write fails, found by swapping one 4a branch at a time:
- The branches that read the family average (3x3 bottleneck, pool) are already right: swaps change KL by .003-.006.
- The 1x1 branch (192 pointwise neurons that read the h1 wave) holds the gap. For readers that one channel cannot
  predict (rho ~ 0), the variance rule makes a large column with a random sign. Training instead uses the loop signal
  to switch off 60 neurons that detect straight lines and textures (window grids, wood grain, vertical bars, fur,
  letters). That is a new gate ("a curve here, so not a straight line"), about 4x stronger than anything the family
  ever wrote to those neurons.
- The plain Bayes write fails for another reason: it shrinks the vote. Fixing the readers' firing rates did not help.
  So the loss is in amplitude further down, where this evidence competes with all other evidence.

## 7b. A second family: the circles (Phase 5, sealed)

The essay's same crush also says "two circle neurons instead of several circle types". Layer 3b has eleven circle
detectors (small dots, eyes, rings, double circles), spread over all four branches. They tile SIZE (6-20 px), as the
curves tile orientation. We applied the k-member recipe with the parameter the family tiles: keep members evenly
spaced in radius (rule fixed before testing), write the Bayes estimate of the rest.

| Circle family (sealed KL) | value |
|---|---|
| removed | .335 |
| derived k=2 (6 and 20 px), no training | .062 |
| naive k=2 (two largest-variance members) | .082 |
| trained k=2 from random | .052 |
| trained k=2 from the derived start | .031 |

Paired: rule vs naive -.020 (se .003); derived vs trained from random +.010 (se .002); derived start vs random start
after training -.021 (se .0014). The same three findings as for the curves: the derived build alone loses to training
at the same size, the derived start beats the random start, and even spacing in the family's own parameter beats the
naive pick.

## 8. Open, not explained

- Why training gates the 60 straight-line and texture neurons with the loop signal. Our guess (it compensates for side
  effects of the coarse neuron's strong writes elsewhere) is NOT tested.
- Why the trained 1x1 write works only as a whole vector. No per-reader law read off it reproduces it.
- Why 10/120/250 degrees beats 50/170/290 degrees for k = 3. Neither variance nor Fisher scores see the difference;
  both rank adjacent triplets worst, correctly.
- The write prefactor ~0.7.
- k = 1 and k >= 2 use different recipes.
- Phase 5 done for a second 3b family (circles, section 7b). Middle and late layers (InceptionV1 4c cars, ConvNeXt last
  layer) are not started.
- Why random-start training fails at k=3 (stuck at the k=1 level) but not at k=10.
- The local crush is not the essay's from-scratch crush.
