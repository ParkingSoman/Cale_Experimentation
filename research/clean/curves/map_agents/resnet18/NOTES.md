# Shared notes: ResNet-18 map (short lines, with your name)
wheels: PREDICTIONS written before wheels_score runs: s4:456 trains pct>.85, bikes/air<.6 (score already seen: held). s4:168 fireboat 554, speedboat 814, steam loco 820 >.8, aircraft <.6. s4:474 steam loco 820 >.8, others <.6. s4:56 canoe 472 and yawl 914 >.7, trains <.5. s4:175 freight car 565 >.85, rest <.6. s4:226 bullet train 466 >.8, other trains <.7.
wheels: s4:226 is a mountain-slope/valley channel, not a train channel. Bullet train (466) uses it as backdrop. Its top parent s3:33 (acts: -2.02 vs controls 0.09/0.02/0.12). Neg pole: small glossy hand-held toys, balls, whistles (round smooth small things).
wheels: s4:456 pos: horizontal bands of parallel rails/keys/rims (trains, piano, steel drum). neg: thin spoke/wing/limb shapes on open background (bicycles, warplane, birds in flight, crab). Reads trains logits. A real train vs bike/plane split lives here, in s4 only so far.
wheels: s4:175 pos: large smooth painted/glazed surface with decoration (ceramic jugs, graffiti boxcars). neg: screens, bottles, helmets. Reads freight car 0.47.
wheels: s4:168 pos: spray/plume/jet of water or steam. neg: smooth round things (car wheel, pill bottle, bubble, nipple). A V against O.
wheels: s4:474 pos: dark black body mass (dogs, siamang, joystick, abaya, fire smoke). neg: tan/light-brown dogs, white bears. A dark-vs-light colour channel.
wheels: s4:56 pos: water surface and underwater view. neg: fine banded or furred land animals, hourglass.
- gate: no gate at stage 2 by the numbers. 15 car classes vs 4130 dog photos: car and dog mean z-min per channel differ by under 0.1 in the top 10 stage 2 channels (tests/gate_scan.py). Stage 2 splits by texture (smooth round rim vs fine fur or mesh: s2:127, s2:109), not by object.
- gate: stage 3 has the gate-shaped channels. Dog (and other furry) faces are the NEGATIVE pole of s3:17, 101, 154, 190, 226, 128, 164. Positive poles differ: printed marks (17), smooth surface (101), lone tall structure (154), repeated pattern (226), vehicle side with wheel (164), body in profile on grass (128). So the dog-face negative is shared; the car part is one channel, s3:164.
- gate: s3:164 is the closest to the essay: wheel on painted body through, faces and daisies turned away. Parent s2:127 (acts: falls 0.58 of 3.13, controls ~0). s2:109 builds s3:101 (0.35 vs 0.05) but not s3:17 or s3:164.
- gate: s3:79 is the opposite sign (dog nose, fur, petals positive). s3:128 is a pose split inside dogs (body in profile vs frontal face).
furbody: coat/body axis in s4 (dog vs cat vs other furry): s4:35 spotted vs plain coat; s4:431 shaggy strands vs sleek; s4:424 bare thick hide vs wiry hair; s4:9 long slender legs vs squat round body; s4:156 broad round furry face vs thin pointed shapes. Dogs/cats do not get one channel: each channel splits all animals by coat or build.
furbody: s4:255 and s4:163 are near-opposites. s4:255 + = flat smooth product close-ups (plastic, cloth, with shop watermark text); - = animals with limbs on a ground (dogs, spiders). s4:163 negative pole shows the same photos as 255's top. s3:17 (printed letters and marks) pushes 255 up and 163 down.
furbody: s3:17 also feeds s4:362 (digital clock digits and white dogs): both = bright shape on a contrasting ground.
furbody: chain s2:71 (one dark round dot) -> s3:222 (dark blotches on pale) -> s4:35 (spots: dalmatian, setter, snow leopard). acts s3:222 -> s4:35 falls 2.30 vs controls ~0.1-0.26.
furbody: carry on cat photos that N calls lynx / weasel / wood rabbit: the channels that push the wrong top class are mostly not coat channels (s4:255, 334, 297 on lynx). Cat-vs-dog is not one gate in this net.
furbody: s4:154 feeds s4:255 by +16 from block 0 (largest sibling I saw). Not read by me.
furbody: acts results: s3:17->s4:362 falls 1.21/20.1 (ctrl ~0.1); s3:71->s4:163 3.36/13.5; s3:30->s4:393 1.68/21.3; s3:128->s4:9 1.75/20.3; s3:76->s4:156 0.54/17.0 (weak); s3:158->s4:431 0.83, ->s4:424 0.51 (weak); s2:71->s3:222 0.32/2.1; s2:6->s3:76 0.38/3.3, ->s3:128 0.12/3.6 (weak).
furbody: every parent I tested fell more than its controls, but the drop is 3 to 20 percent of the firing. One parent never carries a channel; feeds are spread (top feeder 4 to 13 percent).
roadcar: s4:329 = rigid straight-line triangle (rigging, cables, tripod) vs soft fur with small pointed ears. Clean gate. Parent s3:207 (taper on a body) acts 17% of peak. Card exists.
roadcar: s4:486 (pointed sweep vs closed loop) and s4:367 (soft dark cover vs sticking-out part) are antagonists. 367 negative pole = fins, wings, legs.
roadcar: s4:400 (clump of round bumps) fed by both isolated blobs s3:106 and packed blobs s3:140; each acts ~1.2 of 18.
roadcar: s3:2 looked like a parent in trace of photo 17895 but acts on s4:344 is only .11. Trace picks show the photo, not the channel's main parent. Check with acts.
roadcar: Car link is weak. The car classes (wagon, convertible) vote through 32, 461, 269, 400 by shape: smooth sweep, long tapered body, blunt front with two dots. None is a car channel.
roadcar: acts removals are small at s4 (2-17% of peak): many parents share each job.
