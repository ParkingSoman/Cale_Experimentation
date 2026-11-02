# resnet18 cards, agent r18a (last hidden layer, level 0). Tests: tests/r18a_corr.py, r18a_reads.py, r18a_bikedistrict.py

Shared test (r18a_reads.py), all 9 channels: mean h on photos of the channel's 20 strongest reader classes is 2.8-3.8x the mean elsewhere,
and 12-30% of h>3 firings land on those classes (base rate 2%). So the readers and the firings line up, but 70-88% of strong firings
sit outside the 20 reader classes. These are the offshoots, or classes read weakly.

## 179  thin lines running around or out from a hub vs heavy solid dark masses
Poles: + spokes, wheels (unicycle, mountain bike), cannon wheel, cat whiskers, lemon segments, spider legs. - dumbbell, gas mask, lens, screw, dog eye: thick, glossy, solid.
Feeders: bikes arrive through 327/326/286 (feeders-by-photo); lemon, tarantula, tabby arrive through 179's own skip (feeder 179 itself, 4-5.7). Two routes, one channel.
Readers: unicycle, bicycle-built-for-two, tricycle, mountain bike, jinrikisha, Model T; rank 400/512 of logit movement. ablate: KL 0.0011, 85% of it on the 10% most hit photos, tricycle classes lose most.
Prediction: sibling 371 (radiating limbs) co-fires: corr 0.17 (held, weakly). Prediction: 179 opposes 366 (thick glossy cylinder) on pre-activation: corr -0.06 (FAILED: the two poles meet only on a few photos, e.g. dumbbell).
Offshoots: lemon, tarantula, cat nose (own skip route). Sure: 70%.

## 327  a curved loop or arc (handle, hook, flamingo neck)
Poles: + cup handle, padlock shackle, safety pin, teapot spout, flamingo neck. Neg not read. Feeds bikes into 179 (wheel is a loop). Readers: pickup, teapot, whiskey jug, combination lock, coffee mug, tricycle. Sure: 75%. (Not tested beyond sheet and readers.)

## 371  a body with limbs or fur radiating from one centre
Poles: + spider legs, tripod legs, dog muzzle with nose at centre, lionfish fins, harvestman. Readers: tarantula, ant, centipede, scorpion, tick, Persian cat, Shih-Tzu (all radiating-limb or face-with-centre; no noun list needed). Feeders: self (4-5) on spiders, 407 on dog faces, 170/466/485 as helpers: spiders and dog faces arrive by different routes.
Prediction: dog faces come by a route other than the spider route: held (407 vs 371 self). Sure: 65%.

## 161  small sharp thing on a blurred field (macro close-up) vs evenly detailed full-frame scene
Poles: + nail head, ant leg, baseball on blur, ostrich head, fly on grass, fungus cap. - stingray water, pretzel trays, bakery shelves, chain, plate rack, barrel: full-frame regular texture, no blur.
Readers: hip, parking meter, nail, screw, lycaenid, mantis, cicada, bottlecap (small things); negative cliff, megalith, wall clock, maze (big scenes). Rank 8 of 512 logit movement: the one strong reader channel I studied. Feeders: self 3.7 plus 17, 372, 491; negative 284, 182.
Prediction (not run: needs pixel blur measure; I did not do it): blur contrast measure would correlate. Reader classes match by scale (small vs large). Sure: 60%.

## 147  self-lit warm glow on dark vs ordinary daylight scenes
Poles: + match flame, volcano, torch, stage curtain in spotlight, jack-o-lantern, lamp. - shop windows, bottles, barrels, nail: plain daylight lit things.
Readers: theater curtain, lampshade, red wine, torch, lighter, matchstick, stage: all warm-glow things. Sure: 70%.

## 153  soft smooth drooping surface without crisp edges vs crisp small printed or repeated pieces
Poles: + floppy dog ear, hat brim, knitted mitten, woven straw. - orange pile, wine bottles, typewriter keys, bottle cap lettering, uniforms in rows: crisp discrete things.
Readers: cowboy hat, sombrero, bonnet, Rhodesian ridgeback, Labrador, paper towel: soft pendant surfaces (ears, brims). Rank 428: weak reader. Sure: 55%.

## 221  continuous yellow-green organic mass with bumps vs ordered rows of rigid manufactured items
Poles: + custard apple, jackfruit, rapeseed field, dew on leaf, green grass at ballpark. - bookcases, abacus, bottle caps, screws, spoons in rows.
Readers: rapeseed, Granny Smith, tennis ball, indigo bunting, damselfly, cucumber, croquet ball (all yellow-green). The colour is the shared thing, the nouns differ. Corr with 161 0.27 (macro green close-ups) and 366 0.21. Sure: 60%.

## 366  streamlined shiny elongated form with fine speckle or wire hair
Poles: + barracouta, gar, camera lens, dumbbell bar, schnauzer beard, mailbox, printer edge. Neg sheet seen at the top only (not studied). Readers: barracouta, mailbox, terriers (wire-haired), gar, tench. The terriers and fish seem to share fine speckled sheen; sure: 40%.

## 416  dark matte-glossy surface with fine ridges or creases (denim, squash skin, zucchini, echidna spines)
Poles: + jeans, acorn squash, zucchini, echidna spines, boot leather, wool. Readers: jean 0.54, grey whale, acorn squash, zucchini, echidna, leatherback turtle: all rough-skinned dark cloth or skin. Rank 458: weak. Sure: 50%.
