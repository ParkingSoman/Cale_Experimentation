# S6 blind judge: predictions written before each test

## b2 n194 (written 2026-10-21, before synthetic test)
Distinction guess: a single long horizontal boundary between two flat regions (+) vs dense repeated
vertical strokes / fine ornament (-).
P1: a synthetic single horizontal step edge on gray drives it above 50% of max (>2.0); a vertical step edge gives less
than 60% of the horizontal response; a dense vertical grating (period 4-8 px) drives it negative (< -2).
P1 result: FAILED. Horizontal edge 0.01 of max, vertical grating +0.2 (not negative). Best stimulus: a half line
(line that stops in the middle) 0.58 of max. Long lines and edges do nothing.
P2 (before test 2): n194 marks the END of a line. A line that stops drives it to >0.5 of max with the peak within
one cell (4 px) of the endpoint; the same line running through gives <0.15. Short segments (both ends) and L corners
also drive it; dense parallel strokes push it negative (texture of many lines side by side, ends cancelled).
P2 result: CONFIRMED. Line ends 0.3-0.68 of max in all 8 directions, both polarities, widths 1-12 px; full lines
0.01-0.09; dots 0.41; L corner 0.54. Flat edge ~0. It is an end / stop detector.

## b9 n851 (before synthetic test)
Distinction guess: the side (left/right) outline of an upright figure: the strip of background right beside a
vertical object edge (legs, body sides, poles) (+) vs the inside of a large surface or slanted structure (-).
P3: a dark vertical bar (30 px wide, tall) on a light ground drives it above 0.5 of max, with the peak at the bar's
side; the same bar horizontal gives less than half of that; a vertical step edge alone (half-plane) gives less than
the bar; a full-field texture gives a negative max.
P3 result: FAILED as stated. Upright bars, edges and leg silhouettes on flat ground stay near 0 (flat ground itself
-0.23). One stimulus worked: the bar tilted 30 deg off vertical ("/") gave 0.42 of max.
P3b (before test): the neuron wants a near-vertical but tilted contour (10-35 deg off vertical), either lean
(flip on natural images costs only 6%); on a noisy ground the tilted bar reaches >0.5 and the upright bar stays <0.2.
P3b result: CONFIRMED with a caveat. Tilted bars 10-30 deg off vertical on noise ground 0.55-0.61; horizontal
-0.25. Exact vertical: -0.16 pixel-drawn but +0.17 anti-aliased, so "slant beats upright" is partly a pixel
artefact. Safe claim: steep contours (vertical to ~30 deg off) on, horizontal contours and flat ground off.

## b12 n122 (before synthetic test)
Distinction guess: a large empty stretch with nothing in it (sky, plain ground, smooth wall, background around an
object) (+) vs the detailed parts of the object itself (heads, beaks, patterned parts) (-).
P4: a uniform gray field drives it to >0.5 of max; a uniform field with a small textured object in the middle stays
positive far from the object and goes negative on it; a full-field fine noise texture gives <0.2; a bigger empty
area gives a bigger response (empty disc of radius 80 px > radius 30 px, inside textured surround).
P4 result: PARTLY. Uniform fields 0.39-0.49 (just under 0.5). Object on plain ground: -0.28 on the object, +0.35
at the far corners (confirmed). Full-field pixel noise also +0.45-0.62 (prediction <0.2 FAILED: structureless
noise counts as "nothing there"). Bigger empty disc did not give more (failed). Safe claim: "no object structure
here" (+) vs "object structure" (-), not "no texture".

## b15 n1613 (before tests)
Distinction guess: the far bank / shore band of a waterside scene, a horizontal strip of land seen from a distance
with water under it (+) vs one close-up subject filling the frame (dog faces, insects) (-).
P5a: on its top-20 images, painting the part below the peak cell (the water) flat gray drops the response by
>50%; painting the part above the peak cell gray drops it by <30%.
P5b: class means (image max, 20k set): water-edge classes (lakeside, boathouse, dock, canoe, sandbar, seashore) are
>1.5x landscape classes without water (alp, valley, volcano, cliff dwelling).
P5 result: P5a FAILED (masking water below the peak drops only 22%, above 6%: the terrain band itself carries it).
P5b borderline fail (ratio 1.46). Top classes: canoe, paddle, promontory, cliff, boathouse, yawl, dam, valley.
