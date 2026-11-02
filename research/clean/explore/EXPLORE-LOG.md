# Exploration log (Phase 1 of the exploration plan)

Plan: `~/.claude/plans/so-it-seems-that-declarative-ladybug.md`. Lab work, not the paper. Shonu (2026-10-26):
"the first principle is actually exploration". Each move is logged with the fields below. The explanation in words is
written BEFORE any comparison with paper objects (D41, D42). The formal gloss is a separate guess, kept apart.

Network: lucid InceptionV1 (lucent port), ImageNet validation images (cached, 224 crop). Scripts in this folder;
images in `out/`. Pre-ReLU values are in lucid's input scale (pixels 0-255 minus 117), so they are large numbers.

## Neighbourhood 2: 4e:55 (the essay's "cats and cars" unit, which the essay calls a V-shape unit)

### Move 1. The strongest inputs (positive pole)

| Field | Content |
|---|---|
| Looked at | the 24 images (of 10,000 random validation images) with the largest pre-ReLU of 4e:55 at any position, with a box at that position (`s1_poles.py 4e 55`, `out/4e_55_top.png`) |
| Uses | inputs (real images) and the network's forward pass |
| Explanation, in words | Faces seen from the front that are wide at the top (eyes, ears) and narrow to a point at the bottom (the snout): cats, foxes, ferrets, red pandas, a Pomeranian. One lawn-mower front. The box always sits on the face, not the body |
| Formal gloss | the top of the unit's target: a set of inputs where the unit is near its largest value on the data |
| Test that could fail | none yet (this move only proposes) |
| Result | proposal: "a front that tapers to a point" |

### Move 2. The most negative inputs (negative pole)

| Field | Content |
|---|---|
| Looked at | the 24 images with the most negative pre-ReLU (`out/4e_55_bottom.png`) |
| Uses | inputs |
| Explanation, in words | Flat, broad, blunt faces: pugs, mastiffs, boxers, hounds, a French bulldog. Also round things: a lorikeet's head, a spider, an ocarina. The opposite of move 1 is "broad and round in front", not "not a cat" |
| Formal gloss | the bottom of the pre-activation; it names the distinction the unit draws (the essay: "The negative side tells us what distinction matters at this point in the circuit") |
| Test that could fail | Move 1's proposal predicts a negative pole of blunt, round fronts. If the negative pole were unrelated (for example, landscapes or text), the proposal would be in trouble |
| Result | passed. The negative pole is the blunt, round version of the same kind of thing (faces), as the essay says: "round where the V is pointed" |

### Move 3. Weak activations

| Field | Content |
|---|---|
| Looked at | 24 random images whose largest value is 20% to 40% of the unit's largest value (`out/4e_55_weak.png`) |
| Uses | inputs |
| Explanation, in words | Car fronts and hoods (sports car, jeep, police van, truck), a bullet-train nose, an iron's tip, a spoonbill's bill, monkey and lemur faces. All show an outline that converges to a point. Cars appear only here, in the weak band |
| Formal gloss | the outer part of the target (the essay: "a target, with meaning growing out from the center in every direction") |
| Test that could fail | if "tapers to a point" is right, the weak band should be other tapering fronts, not random textures |
| Result | passed, by eye. NOTE: "cats and cars" is what this unit looks like under ImageNet's class labels (cats at the top, cars in the weak band). Under a shape description it is one thing |

Open after moves 1-3: is "tapers to a point" the network's own word, or mine? The next moves ask the network: what
builds the unit (upstream), what reads it (downstream), and a test it can fail (synthetic V and O shapes).

### Move 4. A test that could fail: bare synthetic shapes (`s2_shapes.py 4e 55`, `out/4e_55_shapes.png`)

| Field | Content |
|---|---|
| Looked at | the unit's largest value on plain-background images of one shape each (8 shapes, 3 colour pairs, 4 sizes, 3 jitters) |
| Uses | inputs made by me (no data set), the forward pass |
| Explanation tested | "an outline that is wide at the top and narrows to a point at the bottom" (downward triangle, V strokes above circle and square; circle lowest; direction of the point matters) |
| Formal gloss | a test that can lose: a stated input family with a predicted order of the unit's values |
| Test that could fail | written in the script before the run |
| Result | MOSTLY FAILED. Bare shapes barely drive the unit (best about 5% of the real-face values). Triangles beat circle and square, but the left-pointing triangle scores highest and the downward one lowest of the triangles, and a thin ring equals the downward triangle. So "an outline that tapers to a point" is wrong or far too thin |

What changed: the explanation was too geometric. A second look at move 1: the flat-faced Japanese spaniel is in the
top 24, at its white blaze between dark patches; the ferret, red panda, Arctic fox and tabby cats all have converging
markings on the face. New proposal: "converging lines or markings on a face-like front", not the outline. Lesson for
the formalisation: an explanation has a DOMAIN (here, natural images of fronts); outside it (bare shapes on grey) the
unit is nearly silent, and a test outside the domain says little. The essay's own test (blind maximum) uses examples
written in the unit's domain.

### Move 5. A test that could fail, inside the domain: pointed against flat faces (`s3_groups.py 4e 55`)

| Field | Content |
|---|---|
| Looked at | all 50 validation images of 30 classes chosen BEFORE the run: 20 pointed-snout breeds and foxes (A), 10 flat-faced breeds (B); class mean of the per-image largest value |
| Uses | inputs (class labels chosen by a property of the pictured thing, not by the unit) |
| Explanation tested | "a face-like front that converges to a point, against a broad, blunt one" |
| Formal gloss | a test that can lose: two input families from a description, a predicted order of the unit's means |
| Test that could fail | A mean > B mean; at least 8 of 10 B classes below the median A class; predicted exception Japanese spaniel (flat face, V-shaped blaze) |
| Result | PASSED. A 495, B 274; 9 of 10 B classes below the median A class; the one exception is the Japanese spaniel, as predicted. Surprise: German shepherd (A) scores low (182), with kelpie (271). Their faces are dark with a dark saddle: maybe the V needs light-dark converging markings, not only the snout outline. Open |

### Move 6. Feature visualization with transformation robustness (`s4_fv.py 4e 55`, `out/4e_55_fv.png`)

| Field | Content |
|---|---|
| Looked at | images optimised to raise the unit (averaged over positions, and at the centre) and to lower it, with jitter, scale and rotation (the essay: without "transformation robustness" the method is reward hacked) |
| Uses | weights and the forward pass only: no data set (the image prior is Lucent's) |
| Explanation, in words | Raise: fox and cat faces with pointed snouts, upright ears and white V-shaped markings between dark patches; in the averaged images, green glossy car-hood surfaces are woven between the faces. Lower: round, wrinkled, flat pug-like faces, repeated |
| Formal gloss | the unit's preferred inputs under a stated prior on images (a network-only view of the target's centre) |
| Test that could fail | does the network-only view agree with the data view of moves 1-3? |
| Result | PASSED. It shows both poles found on data: pointed V-marked faces (with car surfaces) and round flat faces. So the poles can be read from the network alone here, with the robustness transforms. (For the inputs rule: moves 1-3 needed data; move 6 got the same answer without data.) |

### Move 7. Upstream: what builds the unit, on two groups of inputs (`s5_upstream.py`, `out/4e_55_upstream.png`)

| Field | Content |
|---|---|
| Looked at | the contribution W[55, j] h_j(x) of each 4d unit j at the unit's best position, averaged over 100 animal-face images (the top of the scan) and over the 100 vehicle images (of 600 in 12 car classes) where 4e:55 is highest; then the top images of each leading 4d unit |
| Uses | weights and inputs |
| Explanation tested | if 4e:55 is one concept, the two groups are built by overlapping 4d units |
| Test that could fail | at least 3 of the two top-10 lists in common (chance about 0.2) |
| Result | PASSED, narrowly (3 in common). One 4d unit, 4d:493, is the largest builder on both groups (586 on faces, 255 on vehicles). 4d:493's own top images: pointed animal faces with V markings (red panda, papillon, ferret, fox, Siamese cat). Faces also use 4d:477, 143, 383 (wolf, dog and small-animal muzzles). Vehicles also use their own units: 4d:352 and 201 (police-car side panels and livery), 4d:445 (pitched roofs, an upside-down V) and 4d:396 (long objects that taper to a point). Votes against: 4d:486, 195 (shaggy, blunt terrier faces) |

Explanation in words: one core route (a pointed, V-marked face unit below) plus an offshoot route for vehicles that
arrives through vehicle units and shape units. Formal gloss (kept apart): a unit's upstream build splits into a CORE
(shared by all its jobs) and OFFSHOOTS (one per job, each with its own builders). The essay's growth picture: "an
offshoot arrives by its own route" (ESSAY-IDEAS I4).

### Move 8. Downstream: what the unit does (`s6_downstream.py`)

| Field | Content |
|---|---|
| Looked at | set the unit's output to 0 at every position; change of each class's log-probability, over the same two groups (classes with mean probability above 1e-3 only; near-zero classes give noise) |
| Uses | inputs and the forward pass (an intervention) |
| Test that could fail | written before the run: on faces it lowers pointed-face classes; on vehicles the effect is small and, if visible, lowers animal classes |
| Result | Faces: PASSED. It lowers red, kit, grey and Arctic fox, weasel, red panda, polecat, lynx, and raises llama, malamute, collie, Samoyed, timber wolf, spaniels: the readers use it as "fox or weasel face, not dog face". Vehicles: FAILED. The effect is as large as on faces (mean KL 0.048 against 0.042), and it lowers limousine, grille, cab, police van and raises jeep, passenger car, minivan, racer, sports car, pickup. It does not lower animal classes |

Line-up (the essay's "You look upstream at how it's built and downstream at what it does, and both line up"):
- faces: lines up. Built from pointed, V-marked face units; used to say "fox-like, not dog-like".
- vehicles: does NOT line up with "V". The unit votes against sports cars (the most V-shaped hoods) and for
  limousines, cabs and grilles (boxy fronts); its vehicle builders are livery and roof units.
So, as an interpreter, I reproduce the essay's V reading for the face job, but not for the vehicle job. The vehicle
job is real (as large an effect at the output) and has its own route in and its own use out. Either the vehicle job
has a description I have not found (one that would also explain "limousine, grille, cab against sports car"), or this
unit carries two jobs that share one builder (4d:493). Not settled; a next move would map the vehicle readers
(which 5a units carry the vehicle effect) and look for the shared description.

## Operations found so far (neighbourhood 2)

| # | Operation | Uses | Output | Can it lose? |
|---|---|---|---|---|
| O1 | strongest inputs of a unit (positive pole), with the position | inputs | a proposal in words | no (proposes) |
| O2 | most negative inputs (negative pole, pre-ReLU) | inputs | the distinction the unit draws | yes: it must fit O1's proposal |
| O3 | weak activations | inputs | the outer part of the target | yes, by eye |
| O4 | a family of inputs made from the description, with a predicted order | made inputs, or classes chosen by a property | pass / fail | yes. LESSON: only inside the unit's domain (bare shapes on grey failed for a reason that says little) |
| O5 | feature visualization with robustness transforms, both signs | weights only | the poles, from the network alone | yes: it must agree with O1, O2 |
| O6 | upstream contributions, split by input group; then O1 on each builder | weights and inputs | core route and offshoot routes | yes: shared builders predicted for one concept |
| O7 | removal (set to 0) and the change at the output, split by input group | inputs, intervention | what the unit is used for, per group | yes |
| O8 | line-up: compare O6 and O7 per group under one description | none new | "lines up" or not, per group | yes. It separated the two jobs of 4e:55 here |

New kinds of operation in this neighbourhood: all eight (it is the first). Stop rule: two neighbourhoods in a row with
no new kind. Next: neighbourhood 1 (the curve family, 3a -> 3b -> 4a).

### Move 9. Split the target by the network's own route (no class labels) (`s7_split.py`, `s8_routes.py`)

| Field | Content |
|---|---|
| Looked at | 3000 positions drawn from the unit's target (probability proportional to its output) over 3000 images; each split by its lead builder (the unit below with the largest vote); the top images of each large route |
| Uses | weights and inputs |
| Explanation, in words | 4e:55's routes: 4d:493 = pointed animal faces (27% of the target); 4d:352 = vehicle bodies (8%); 4d:61 and 4d:304 = image borders and corners (a padding effect). So the network's own split finds "faces" and "vehicles" with no labels |
| Formal gloss | the route of a unit at an input = argmax over builders of the vote; the target splits into routes |
| Test that could fail | the route split should recover the face / vehicle split of move 7 |
| Result | PASSED. A first try (`s7_split.py`: k-means with k = 2 on the build vectors) split "faces" from "the broad weak band" instead; the lead builder needs no chosen k and is a one-unit name of the route |

### Move 10. Use, measured at the readers (`s8_routes.py`)

Two failed forms first (logged because they shape the definition):
- use = the effect on the output distribution, M dy/dh_u (`s7_split.py ... output`): single use vectors are nearly
  orthogonal even inside one route (cosine 0.05 to the route mean): the output effect is dominated by which class the
  image is, so it records the image, not the unit's job. Worked by hand on small circuits: with this use, the "V"
  case (one unit read by one reader on cats and cars) scores as packed, because the winning class differs.
- distance correlation between single build and single use vectors: about 0, inside the shuffled null (too noisy).

| Field | Content |
|---|---|
| Looked at | the gated reads of the next block: for each reader r that reads the unit through a 1x1 conv, W_ru (reader rescaled to unit norm) if r is on at that position, else 0; route means of this vector; packing index pi = 1 - |sum_g w_g m_g|^2 / (sum_g w_g |m_g|)^2 (0 when all routes feed the same readers, up to strength) |
| Uses | weights and inputs |
| Explanation, in words | 4e:55: the face route and the vehicle route feed mostly the same readers (cosine 0.94; top reader 5a:360 on every route); pi 0.047 (shuffled-route null at most 0.001). Curve unit 3b:379: routes = round edges, spirals and clock faces, and image borders; same readers (cosine 0.93 to 0.99); pi 0.011. Car unit 4c:447: routes = car fronts and wheel arches, car sides and doors, an offshoot of boxy panelled machines (espresso maker, parking meter, gas pump, safe), and a weak band; same readers (0.95 to 1.0); pi 0.012 |
| Formal gloss | the use of a unit at an input = its gated reads; packing = routes with different mean uses |
| Test that could fail | predicted: 4e:55 above its null; the curve unit near its null |
| Result | 4e:55 PASSED (about 4 times the curve unit and the car unit). The curve unit is above its null too (0.011 against 0.000), so "near the null" FAILED in the strict sense: every unit has a small route-dependence. The scale matters, not zero |

Reading: locally (one layer up) 4e:55 is mostly one job: both routes reach the same readers. Its two different effects
at the output (move 8: fox against grille) arise further downstream, through readers that are on in different
contexts. The car unit shows the essay's growth picture: a core (car parts) and an offshoot (boxy panelled machines)
used like the core.

## Neighbourhoods 1 and 3, and the stop rule

| Neighbourhood | Moves used | New kinds of move |
|---|---|---|
| 2: 4e:55 | O1 to O10 | all ten (first neighbourhood) |
| 1: the curve unit 3b:379 | O9, O10 | none |
| 3: the car unit 4c:447 (essay's car neuron) | O1, O2, O3, O9, O10 | none |

Stop rule met (two neighbourhoods in a row with no new kind). Honest limit: neighbourhood 1 was explored thinly
(the curve family is known from the earlier curve work, `CURVES-LOG.md`).

## Operations table (final for Phase 1)

| # | Operation | Uses | Output | Needs real images? |
|---|---|---|---|---|
| O1 | strongest inputs (positive pole) | inputs | a proposal in words | yes, or O5 |
| O2 | most negative inputs (negative pole, pre-ReLU) | inputs | the distinction drawn | yes, or O5 |
| O3 | weak activations | inputs | the outer target | yes |
| O4 | a family of inputs from the description, predicted order | made inputs or chosen classes | pass / fail (inside the domain only) | partly |
| O5 | feature visualization with robustness transforms | weights only | the poles | no |
| O6 | upstream contributions by input group | weights, inputs | core and offshoot builders | yes |
| O7 | removal and the output change by input group | inputs, intervention | the use at the output, by group | yes |
| O8 | line-up of O6 and O7 per group | none new | lines up or not | (as O6, O7) |
| O9 | split the target by the lead builder | weights, inputs | the network's own routes | yes (a law on inputs) |
| O10 | gated reads by route; packing index | weights, inputs | route-dependence of use | yes (a law on inputs) |

For the inputs decision (Shonu, after Phase 1): every move except O5 needs a law on inputs. O5 found the same poles
as O1, O2 for 4e:55 without data. O9 and O10, the moves the definition rests on, need a law on inputs (the target).

### Move 11. The scale of the packing index (`s9_scale.py 4e 55 8`), and the shared reader 5a:360

| Field | Content |
|---|---|
| Looked at | (a) 5a:360, the top reader of every route of 4e:55, with `s8_routes.py`; (b) planted two-job units: a virtual unit that is one real 4e unit on its target and another on its own, with their real gated reads; (c) 9 units of 4e in the trained network and in a random-init InceptionV1 |
| Test that could fail | planted pairs at least 5 times the single-unit median; random-init units higher than trained |
| Result | (a) 5a:360: pi 0.069, above the 4e units; its routes are lemons and round yellow fruit, white fluffy faces, dog and fox faces, skin. (b) planted median 0.49, 13 times the trained single-unit median 0.037: PASSED. 4e:55 (0.043) is typical for its block. (c) FAILED, and informative: random-init units score pi near 0.001, LOWER than trained. In a random network the readers ignore the route, so every unit looks like one job |

### Move 12. Is the unit READ? (`s10_read.py 4e 55 8`)

| Field | Content |
|---|---|
| Looked at | read index lambda: the extra signed read (sum of rescaled read weights of the readers that are on) where the unit is on, against anywhere, divided by the total read weight |
| Uses | weights and inputs |
| Explanation, in words | the essay's own definition of clean: "built to be read". A unit is read when its positive readers turn on with it and its negative readers stay off |
| Test that could fail | trained lambda clearly above 0; random-init lambda near 0 |
| Result | PASSED. Trained: all 9 units positive (+0.029 to +0.117, median +0.046). Random init: all between -0.000 and +0.001. (The plain correlation of h_u with the signed read was noisier: random-init units reached +0.5) |

## Where the exploration ends: the definition (paper A.6, Definitions D43 to D45)

A unit is clean when it has ONE JOB (packing index pi = 0: every route by which it is built feeds the same readers)
and is READ. The read part has two forms (paper D45 item 5): the weak form lambda > 0, and the agreement form xi > 0
(move 13), which is the one that separated trained from random networks in Part C (H-agree). Both parts are needed: a random network passes ONE JOB and fails READ; a planted
two-job unit passes READ and fails ONE JOB. Checks: `curves/model/t47_routes.py` (planted circuits, Theorem T48);
`curves/pc_route.py` (Hypothesis H-route on InceptionV1).

### Move 13. The excess read (`s11_excess.py 4e 10 1000`, `s11_excess.py 3b 10 1000`)

Why: the Part C measurement of H-route (`curves/pc_route.py`) found 0.767 of random-init units with lambda > 0
(threshold 3/4 failed). Lemma T46b explains it: a reader always responds a little to the unit's own vote. So the
question becomes whether the readers' OTHER votes agree with the unit.

| Field | Content |
|---|---|
| Looked at | xi = lambda - lambda_ind, with lambda_ind computed with each reader's rest (pre-activation minus the unit's vote) taken from an independent input (a first version paired each value with its own rest in the baseline; fixed before the numbers below) |
| Test that could fail | trained xi > 0 for at least 9 in 10; random-init share at most 3 in 4, with much smaller sizes |
| Result | PASSED. 4e: trained 9 of 10 positive (median +0.0125), random 6 of 10 (median size 0.0002). 3b: trained 10 of 10 (median +0.0212), random 3 of 10 (median size 0.0053). In the trained network about 60% of lambda is the unit's own vote (lambda_ind median +0.033 of +0.047 in 4e) |

Consequence worked out on paper (Proposition T50): T43's detector units are never on together, so their readers'
other votes never agree with any one of them, and xi < 0. A reader that ORs alternatives that are never on together
fails the agreement read. Real siblings usually co-fire weakly (the essay's "analogizers"), so this may not bite in
trained networks; it is a choice for Shonu (card 3-7).
