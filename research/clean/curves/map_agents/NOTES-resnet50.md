# Shared notes: resnet50 last layer (append short lines, with your agent name)
r50a: 364 and 972 are one family (corr 0.58, others 0 to 0.2). 972 = body/arm on a bar (gymnastics core). 364 = wider: any extremity, ball-sport arms, and furry dog paws/long-haired dogs by their own route.
r50a: 257 = dense parallel stalks/husks (corn, wheat, reeds). Bittern, chameleon, ringlet fire because of the reeds, not the animal. Neg = smooth blur.
r50a: 1699 = flat blades/petals fanning from a centre (pinwheel, rapeseed, daisy, umbrella). Not the negative pole of 364.
r50a: At level 0 the feeders of a channel are mostly its own skip path (257: 2-5 of ~6). Use --level 1 for real routes.
r50a: Pairwise corr of 9 level-0 channels is about 0 except 364/972. Top 2% fires ~2.0 vs mean 0.4: long tail, on for >85% of P. evidence.py `corr` and `feeders` are slow when 3 agents run (120 s timeout).
r50a: claim_error: gate regions are graded not sharp. Pair 364+972, theta 3.0: mass .02, eps_dist .91; theta 1.0: mass .37, eps .35; theta .6: mass .65, eps .13. kl_drop of each is tiny (<6e-4).
r50c: last layer channels are on for ~90% of P; the strong tail is small. Feeders at level 0 are the channel itself (identity skip); use feeders-by-photo only to see side routes.
r50c: 214 = curl (wiry dog hair, corkscrew, mantis leg), 1113 = sleek short coat vs fluffy fur (214 and 1113 are one coat axis), 1254 round solid vs fine mesh, 1258 big flat panel vs sharp points (freight car), 1256 curved banded cylinder (barrel), 88 thin dark line on sky, 1934 coarse straw strands. Cards in cards/resnet50_r50c.md.
r50c: claim_error on single channels with own-gate: eps_dist 0.8-0.95 (most activity sits outside the strong region), kl_drop ~1e-4 per channel. Ablating all 8 gives KL 0.001, top-1 agreement 0.991. Off-region "off" claims fail; the strong tail still carries 93% of the KL of an ablation (10% most-hit photos).
- r50b: ALL 11 channels read at level 0 fire on 86-98% of P; top 5% of photos hold ~20% of a channel's activity, top 20% ~50% (same for 128 random channels). No "off outside region".
- r50b: weak firings (bottom half of h) have rank-corr 0.0-0.14 with the centred top-reader logit; top 20% 0.11-0.43. Graded evidence, mostly read in the tail.
- r50b: level-1 feeders of any last-layer channel = itself (~0.88 of input x grad): last block is near pass-through. Use level 2 for circuits.
- r50b: 734,342 = pale soft coat vs dark dense fur (test held). 1852,1994 = thin straight lines/rods (shoji, redshank legs). 1882 = glossy curved highlight. 432 = soft fine filaments (not stiff spines: porcupine test failed). 969 = flat slab over gap vs thin rods. 1350 = small bright repeated cells. Cards: cards/resnet50_r50b.md
- r50b: gate claims (sum of own members > p80/p90): eps_dist 0.53-0.76, kl_cut ~0.000-0.001, own_inside ~0.001-0.002. Districts of 2-4 channels carry no decision weight alone.
