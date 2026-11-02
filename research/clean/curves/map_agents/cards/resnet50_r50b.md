# resnet50 cards, agent r50b (last hidden layer, level 0; 11 channels studied)
Common facts (all 11): "on" (h>1% of max) on 86-98% of P; top 5% of photos hold ~20% of the activity, top 20% hold ~50% (the
typical channel is the same, measured on 128 channels). Level-1 feeders: the channel itself carries ~0.88 of the input x gradient
(last block mostly passes the residual stream through), so upstream reading needs level 2+ (not done). Weak firings (bottom 50% of
h) have rank-corr 0.0-0.14 with the centred logit of the channel's top reader class; top 20% gives 0.11-0.43 (r50b_dose.py).

## 432
Verb: soft fine filaments radiating from a centre (thistle floss, daisy petals, bristled legs) against smooth round/concentric hubs.
Neg pole: fans, compass, bubble, pinwheel, round smooth things. Reads: pineapple, cardoon, butterflies (positive); small weights.
Feeds: not read (level 1 is itself). Prediction: fires on porcupine/sea urchin as on cardoon. RESULT: partly failed (porcupine 0.56 vs
cardoon 3.9, sea urchin 0.9). Refined: soft/feathery filaments, not stiff spines. Offshoots: lycaenid/ringlet wing surfaces with scale
fuzz. Sure: medium.

## 734 (+342, neighbour r=0.57)
Verb: broad pale smooth soft surface against dark dense fur or plumage. Neg pole: dark coats (Newfoundland, water ouzel, lynx fur,
black poodle, ping-pong ball of black fuzz, cuirass). Reads: Labrador, stork, setter, spoonbill, Samoyed. Prediction: pale-coat classes
(Samoyed, Great Pyrenees, ice bear) high, dark-coat dogs (Newfoundland, curly-coated, Kerry blue, black swan) near 0. HELD: 1.1-2.2
vs 0.05-0.26. Offshoots: hog skin, conch, apple flesh, egret wing, flamingo, spoonbill (pale pink/white soft). 342 is the fluffy-long-
hair version (Angora, Maltese, sheepdog). Sure: high for the pale/dark axis.

## 969
Verb: a broad flat slab (brim, table top, lid) edging over a gap, against thin rods and small bumpy items. Neg pole: bannister,
tripod, parallel bars, screwdriver, fire-screen bars, armour plates. Reads: sombrero, bonnet, birdhouse, barrow, dining table, mortarboard.
Top photos: hat brim over eyes, table edge with one leg, overhang. Prediction not tested by code (no class test run). Sure: medium.

## 1852 (+1994, r=0.51)
Verb: thin straight light lines crossing in a grid on a pale ground, against bold dark slats and rich-coloured blobs. Neg pole: worm
fence, maraca, butternut squash, feather boa, hourglass. Reads: shoji 0.38, centipede, walking stick, redshank (thin legs).
Prediction: shoji and window screen high, worm fence/stone wall/cardoon near 0. HELD: 5.0 and 1.05 vs 0.29/0.13/0.18; picket fence
0.29 (thick slats) also low, as predicted. Offshoot: walking stick, spindle. 1994 is the same family in warm orange wood: shoji
frames, crib bar, redshank legs, eft, plunger handle: "thin straight orange rod". Sure: high for 1852, medium for 1994.

## 1882
Verb: curved glossy mirror-like surface with a bright highlight, against matte patterned surfaces. Neg pole not viewed (sheet came
late; reader negatives are matte: wool mitten, pot, diaper). Reads: loupe, car mirror, parking meter, hourglass. Top: bubble, lens,
mirror, disco ball, acorn squash skin, cassette window. Prediction: bubble, loupe, car mirror, parking meter high vs spoon,
cardoon, Samoyed low. HELD: 1.2-1.7 vs 0.49/0.12/0.06. Sure: medium-high.

## 544
Verb: soft irregular pitted matte surface (crumb, bark, ice-cream scoop) against smooth shapes. Reads: chocolate sauce, ice cream,
cardigan, bakery. Offshoots: orange peel, stinkhorn, snail shell stone. No prediction run. Sure: medium-low.

## 1350
Verb: small repeated bright cells on a grid or scale pattern (scales, 7-segment digits, slot-machine screens, turnstile, feather eye
spots, glowing dots on black). Reads: green mamba/snake, tiger beetle, police van, digital clock, modem, slot. Mixed readers show
two routes: scale patterns (snakes, peacock) and lit-digit machines. Offshoots: police van yellow panel, swab. Sure: medium.

## 1895
Verb: glossy translucent smooth yellowish liquids and rounded gleaming things (consomme, lotion, corn kernels). Weak neg pole (mean -1
at best, diffuse). Reads: sunscreen, eggnog, consomme. Not confirmed. Sure: low.

## 911 (neighbour of 432, r=0.50) and 1994 (neighbour of 1852)
911: thin jointed limbs/legs against a blurred ground (insect, ant, mantis legs). Its correlation with 432 is partly photo
co-occurrence (bee on daisy), not one verb. Sure: medium. 1994: see 1852.
