# resnet18 cards, agent r18b (level 0, last hidden layer)

## 273: light that a thing gives off, against darkness (vs light that bounces off a lit object)
- Poles: + glowing blobs on dark (jack-o'-lantern faces, torch flame, digital clock digits, stage beams, matchstick, jellyfish glow). - matte or shiny mechanical parts lit from outside (brake disc, camera body, cannon wheel, lens).
- Feeds: its own skip dominates (273 contributes 2-10 of each top photo's input); next 147 (self-lit glow, r18a), 137, 401. Corr 273-147 0.36, 273-401 0.29.
- Reads: no clean readers (candle, jack-o'-lantern, scoreboard, digital clock, spotlight, stage, but also platypus, axolotl). Own share of decision 0.4 of max, rank 340.
- Prediction: candle, spotlight, volcano, jack-o'-lantern score higher than table lamp, lampshade, lighter, monitor, TV. Result: HALF. Candle (pct 0.90) and jack-o'-lantern (0.98) high; spotlight 0.63, volcano 0.65 no better than near misses (TV 0.77, monitor 0.73). Daylight volcano/spotlight photos lack the dark surround. So "emission" is not enough; "emission surrounded by dark" is closer. Pixel check: corr of 273 with dark fraction of photo 0.43, with mean luminance -0.40.
- Offshoots: nematode (green blurred blobs), oscilloscope, jellyfish (bright dots in dark). Same verb, different route.
- Sure: medium.

## 401: small round pits or dots tiled over a surface, and open blue water (vs long straight lines: fences, book spines, bars)
- Poles: + jellyfish spots, golf ball dimples, puffer spots, gas-mask vents, radio-telescope dimples, AND scuba divers and sharks in blue water. - chain-link, shelves of spines, bench railings, bread seams: line structure.
- Feeds (level 1, by photo): two routes. 39/132/106/171 on blue-water photos (jellyfish, divers, shark, anemone fish). 157/4/117 on dimples and rings (golf ball, conch, gas mask, radio telescope). Level 1 sheets not read (tool timed out).
- Prediction: golf ball, jellyfish, puffer score higher than ping-pong ball, tennis ball, honeycomb, brain coral, scuba diver. Result: FAILED in part. Scuba diver 0.95 pct equals jellyfish 0.95; golf ball only 0.74, honeycomb 0.57, brain coral 0.76. Water is the strong core; the dimples facet is weaker and does not extend to other tilings.
- Offshoots: electric guitar, trombone, nautilus (feeder 4, 117).
- Sure: medium-low.

## 120: a band that curves round and keeps running parallel to itself (vs scattered small separate items)
- Poles: + maze hedges, coils, concentric rings, iPod click wheel, nematode and a curving buffalo back, elephant trunk folds. - shark teeth row, scale ticks, glassware shelves, rivets, dumbbell grip: many small discrete repeats.
- Reads: maze (0.45) top, then big grazing animals (sorrel, buffalo, tusker, warthog, camel). The animal facet fits "thick curved body mass with ridged fur", an offshoot.
- Prediction: maze 646, brain coral 109, water buffalo 346, ... high; golf ball, ping-pong low. Held: maze 0.95, buffalo 0.98, golf 0.39, ping-pong 0.48. Brain coral 0.78 (curved ridges but weaker).
- Not colour, not brightness (all |corr| < 0.07).
- Sure: medium.

## 222: many same-size touching cells tiled across a close-up (vs one separate object on plain ground)
- Poles: + keyboard keys, honeycomb, corn kernels, thimble pits, snake scales, photocopier buttons. - screws, key rings, a single face, crib, candy-pattern cloth: isolated items.
- Same family as 401 dimples? Corr 222-401 is -0.02, so NOT the same channel; two tilings with different routes. Own share of decision is rank 439: almost unread.
- Prediction (not tested by score): keyboards and honeycomb beat a single-bolt photo. Sheet looked consistent; no number run.
- Sure: medium-low.

## 224: warm yellow-tan mass of soft pieces (vs cool dark grey or purple hard items)
- Poles: + carbonara, dough, bakery, corn, wooden plane, wine bottles, basket. - paint brush, lipstick, desktop computer, scoreboard, artichoke, bear face.
- Pixel check: corr with red-minus-blue 0.34, saturation 0.21. So it is partly a colour channel. Facet "soft/pile" (spaghetti, eggnog) vs colour not separated.
- Reads: plane 0.40, mousetrap, hamper, quill, apiary, crate (loose).
- Prediction: warm tone dominates over texture: failed to confirm: corr only 0.34, a lot lies elsewhere.
- Sure: low.

## 404: smooth plain pale flat surface with an edge, as in bathrooms and white furniture (vs saturated outdoor clutter with rope and stripe lines)
- Poles: + washbasin, plunger, medicine chest, wardrobe, shoji, bathtub. - stop sign, chain, sails, maypole, coral reef.
- Corr with 224: 0.23 (both smooth, plain). With 120: -0.17.
- Sure: low-medium. Not tested beyond sheets and pixel stats (lum 0.13, sat -0.02).
